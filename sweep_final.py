"""
sweep_final.py -- fair three-way comparison
    Sharon (vanilla)  vs  Design 1 (weighted round ordering)  vs  Design 2 (dynamic matching)

What is different from sweep_multi_shard_gem.py
  * Never touches config/params.json: every run gets its own params file, so runs can go in parallel (--workers).
  * Information cost: Designs 1 and 2 are run with info_delay_rounds k = 0, 1, 2 (counts are k rounds stale).
  * Latency is measured at the end of the pair's consensus; warm-up transactions are excluded.
  * Transaction arrivals are identical across modes (private RNG stream per node), so seeds are truly paired.
  * Extra metrics: p95, pair-averaged, hot/cold-pair latency, meeting gaps, epoch/round diagnostics,
    pending-at-end, rejected, conservation residual.
  * Reports a 95% CI for every paired difference.

Stages
  tune : seeds 1-5, n=10, skews 1.0 & 2.0 -- choose alpha, ablate the three Design-2 changes
  main : seeds 6-15, n in {4,6,10}, skews {0,1,2} -- the numbers you report (use the alpha chosen in 'tune')

Usage
  python sweep_final.py --stage tune --workers 4
  python sweep_final.py --stage main --alpha 0.1 --workers 4
  python sweep_final.py --stage main --summary
  python sweep_final.py --stage main --n 4,10 --skews 0,2 --seeds 6-8      # any subset / override
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_PARAMS_FILE = "config/params.json"
TMP_DIR = "sweep_tmp"

METRICS = {
    "blockchain_length": r"Length of Blockchain = (\S+)",
    "cross_included": r"Total no of cross-shard transactions included in Blockchain = (\S+)",
    "cross_processed": r"Total no of cross-shard transactions processed = (\S+)",
    "cross_generated": r"Total no of cross-shard transactions generated = (\S+)",
    "accepted_tps": r"Accepted TPS = (\S+)",
    "measured_txs": r"Measured cross-shard txs \(after warm-up\) = (\S+)",
    "avg_latency": r"Average cross-shard tx confirmation latency = (\S+)",
    "max_latency": r"Max cross-shard tx confirmation latency = (\S+)",
    "p95_latency": r"P95 cross-shard tx confirmation latency = (\S+)",
    "pair_avg_latency": r"Pair-averaged cross-shard latency = (\S+)",
    "hot_latency": r"Hot-pair \(top 20% by traffic\) avg latency = (\S+)",
    "cold_latency": r"Cold-pair \(remaining 80%\) avg latency = (\S+)",
    "num_rounds": r"Number of rounds executed = (\S+)",
    "mean_round_duration": r"Mean round duration = (\S+)",
    "mean_active_pairs": r"Mean active pairs per round = (\S+)",
    "mean_epoch_duration": r"Mean epoch duration = (\S+)",
    "max_gap": r"Max meeting gap \(any pair\) = (\S+)",
    "mean_pair_max_gap": r"Mean per-pair max meeting gap = (\S+)",
    "pending_end": r"Pending cross-shard txs at end = (\S+)",
    "pending_mean_age": r"Mean age of pending cross-shard txs = (\S+)",
    "pending_max_age": r"Max age of pending cross-shard txs = (\S+)",
    "rejected": r"Rejected cross-shard txs = (\S+)",
    "late_completions": r"Late completions \(after sim end\) = (\S+)",
    "residual": r"Conservation residual \(generated - processed - rejected - pending\) = (\S+)",
    "total_volume": r"Total accepted cross-shard tx value \(volume\) = (\S+)",
    "busiest_pair_pct": r"busiest pair carries ([\d.]+)%",
}
HEADER = ["num_shards", "traffic_skew", "mode", "seed"] + list(METRICS.keys())


class Mode:
    def __init__(self, label, weighted, dynamic, alpha, delay, extras=None):
        self.label, self.weighted, self.dynamic = label, weighted, dynamic
        self.alpha, self.delay, self.extras = alpha, delay, extras or {}


def make_modes(stage, alpha):
    vanilla = Mode("vanilla", False, False, None, 0)
    if stage == "tune":
        modes = [vanilla, Mode("weighted_k0", True, False, None, 0)]
        for a in (0.01, 0.1, 1.0, 5.0):
            modes.append(Mode(f"dyn_a{a}", False, True, a, 0))
        # ablations of the three changes made to Design 2 (alpha fixed at 0.1)
        modes.append(Mode("dyn_a0.1_maxcard", False, True, 0.1, 0,
                          {"dynamic_edge_filter": False, "dynamic_maxcardinality": True}))   # original pre-change version
        modes.append(Mode("dyn_a0.1_filter_maxcard", False, True, 0.1, 0,
                          {"dynamic_edge_filter": True, "dynamic_maxcardinality": True}))
        modes.append(Mode("dyn_a0.1_gapcap", False, True, 0.1, 0,
                          {"dynamic_gap_cap_rounds": "epoch"}))                               # soft anti-starvation cap
        return modes
    modes = [vanilla]
    for k in (0, 1, 2):
        modes.append(Mode(f"weighted_k{k}", True, False, None, k))
        modes.append(Mode(f"dynamic_k{k}", False, True, alpha, k))
    return modes


# ---------------------------------------------------------------------------------------------
def load_base():
    with open(BASE_PARAMS_FILE) as f:
        return json.load(f)


def parse_list(s, cast):
    return [cast(x) for x in s.split(",") if x != ""]


def parse_seeds(s):
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return parse_list(s, int)


def run_one(base, args, n, skew, mode, seed):
    params = dict(base)
    params.update({
        "num_nodes": args.nodes, "simulation_time": args.sim_time, "num_shards": n,
        "sharon_enabled": True, "weighted_scheduling": mode.weighted, "dynamic_matching": mode.dynamic,
        "elastic_capacity_enabled": not args.no_elastic, "traffic_skew": skew, "seed": seed,
        "verbose": 0, "info_delay_rounds": mode.delay, "warmup_time": args.warmup,
    })
    for key in ("dynamic_edge_filter", "dynamic_maxcardinality", "dynamic_gap_cap_rounds", "aging_alpha"):
        params.pop(key, None)
    if mode.alpha is not None:
        params["aging_alpha"] = mode.alpha
    for key, val in mode.extras.items():
        params[key] = n if val == "epoch" else val

    uid = hashlib.md5(f"{args.stage}|{n}|{skew}|{mode.label}|{seed}|{args.tag}".encode()).hexdigest()[:12]
    os.makedirs(TMP_DIR, exist_ok=True)
    pfile = os.path.join(TMP_DIR, f"{uid}.json")
    with open(pfile, "w") as f:
        json.dump(params, f, indent=2)

    dir_suffix = f"sweep_final{args.tag}/{args.stage}/n{n}_{mode.label}/skew{skew}_seed{seed}"
    res = subprocess.run([sys.executable, "simulate.py", dir_suffix, pfile], capture_output=True, text=True)
    if res.returncode != 0:
        return (n, skew, mode, seed), None, (res.stdout[-800:] + res.stderr[-1500:])
    log_dir = os.path.join("simulation_logs", dir_suffix)
    logs = [x for x in os.listdir(log_dir) if x.endswith(".log")] if os.path.isdir(log_dir) else []
    if not logs:
        return (n, skew, mode, seed), None, f"no log file in {log_dir}"
    return (n, skew, mode, seed), os.path.join(log_dir, logs[0]), ""


def parse_log(path):
    with open(path) as fh:
        return parse_log_text(fh.read())


def parse_log_text(content):
    row = {}
    for key, pat in METRICS.items():
        m = re.search(pat, content)
        v = m.group(1) if m else ""
        row[key] = "" if v in ("N/A", "nan") else v
    return row


def already_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("avg_latency"):
                    done.add((int(r["num_shards"]), float(r["traffic_skew"]), r["mode"], int(r["seed"])))
    return done


# ---------------------------------------------------------------------------------------------
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
       2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
       2.045, 2.042]


def ci95(values):
    """mean, half-width of the 95% t-interval"""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = statistics.mean(values)
    if n < 2:
        return m, float("nan")
    t = T95[n - 2] if n - 1 <= 30 else 1.96
    return m, t * statistics.stdev(values) / math.sqrt(n)


def fl(x):
    try:
        return float(x) if x not in ("", None) else None
    except ValueError:
        return None


def paired_pct(rows, a_mode, b_mode, key, n, skew):
    """per-seed % change of b relative to a (negative = b lower/better), paired by seed"""
    A = {int(r["seed"]): fl(r[key]) for r in rows if int(r["num_shards"]) == n and float(r["traffic_skew"]) == skew and r["mode"] == a_mode}
    B = {int(r["seed"]): fl(r[key]) for r in rows if int(r["num_shards"]) == n and float(r["traffic_skew"]) == skew and r["mode"] == b_mode}
    out = []
    for s in sorted(set(A) & set(B)):
        if A[s] and B[s] is not None and A[s] > 0:
            out.append((B[s] - A[s]) / A[s] * 100)
    return out


def mean_of(rows, mode, key, n, skew):
    vals = [fl(r[key]) for r in rows if int(r["num_shards"]) == n and float(r["traffic_skew"]) == skew and r["mode"] == mode]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else float("nan")


def fmt_ci(vals):
    m, h = ci95(vals)
    if math.isnan(m):
        return "      n/a"
    return f"{m:+6.1f}% ±{h:4.1f}" if not math.isnan(h) else f"{m:+6.1f}%  (1 seed)"


def summarize(path):
    if not os.path.exists(path):
        print(f"No results file {path}")
        return
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("avg_latency")]
    ns = sorted({int(r["num_shards"]) for r in rows})
    skews = sorted({float(r["traffic_skew"]) for r in rows})
    modes = []
    for r in rows:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    others = [m for m in modes if m != "vanilla"]

    print("\n=== PAIRED CHANGE VS VANILLA SHARON (negative = better), mean over seeds ±95% CI ===")
    print(f"{'n':>3} {'skew':>4} {'mode':<24} {'seeds':>5} {'vanilla':>8} {'mode':>8}  {'avg latency':<16} {'p95':<16} {'pair-avg':<16} {'hot':<16} {'cold':<16}")
    for n in ns:
        for sk in skews:
            for m in others:
                avg = paired_pct(rows, "vanilla", m, "avg_latency", n, sk)
                if not avg:
                    continue
                print(f"{n:3d} {sk:4.1f} {m:<24} {len(avg):5d} {mean_of(rows,'vanilla','avg_latency',n,sk):8.2f} {mean_of(rows,m,'avg_latency',n,sk):8.2f}  "
                      f"{fmt_ci(avg):<16} {fmt_ci(paired_pct(rows,'vanilla',m,'p95_latency',n,sk)):<16} "
                      f"{fmt_ci(paired_pct(rows,'vanilla',m,'pair_avg_latency',n,sk)):<16} "
                      f"{fmt_ci(paired_pct(rows,'vanilla',m,'hot_latency',n,sk)):<16} {fmt_ci(paired_pct(rows,'vanilla',m,'cold_latency',n,sk)):<16}")

    print("\n=== DESIGN 2 VS DESIGN 1 AT THE SAME INFORMATION DELAY (paired by seed) ===")
    for n in ns:
        for sk in skews:
            for k in (0, 1, 2):
                a, b = f"weighted_k{k}", f"dynamic_k{k}"
                if a in modes and b in modes:
                    v = paired_pct(rows, a, b, "avg_latency", n, sk)
                    if v:
                        print(f"n={n:2d} skew={sk:3.1f} k={k}:  dynamic vs weighted avg latency {fmt_ci(v)}   ({len(v)} seeds)")

    print("\n=== MECHANISM / FAIRNESS DIAGNOSTICS (means over seeds) ===")
    print(f"{'n':>3} {'skew':>4} {'mode':<24} {'epoch dur':>9} {'round dur':>9} {'act pairs':>9} {'max gap':>8} {'pair maxgap':>11} {'pending':>7} {'rejected':>8} {'residual':>8} {'tps':>6}")
    for n in ns:
        for sk in skews:
            for m in modes:
                if not any(int(r["num_shards"]) == n and float(r["traffic_skew"]) == sk and r["mode"] == m for r in rows):
                    continue
                g = lambda k: mean_of(rows, m, k, n, sk)
                print(f"{n:3d} {sk:4.1f} {m:<24} {g('mean_epoch_duration'):9.2f} {g('mean_round_duration'):9.2f} {g('mean_active_pairs'):9.2f} "
                      f"{g('max_gap'):8.1f} {g('mean_pair_max_gap'):11.1f} {g('pending_end'):7.1f} {g('rejected'):8.1f} {g('residual'):8.1f} {g('accepted_tps'):6.2f}")
    print("\nNegative conservation residual = bug. epoch dur is n/a for dynamic modes (no fixed epoch).")
    print("Read the diagnostics before the headline: if a mode's epoch/round durations differ from vanilla's, "
          "part of its gain is a change in round cost, not in whom it matches.")


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "main"], default="main")
    ap.add_argument("--alpha", type=float, default=0.1, help="aging alpha for Design 2 in the main stage")
    ap.add_argument("--n", default=None, help="comma list of shard counts (override)")
    ap.add_argument("--skews", default=None, help="comma list of traffic skews (override)")
    ap.add_argument("--seeds", default=None, help="e.g. 6-15 or 1,2,3 (override)")
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--sim-time", type=int, default=1000)
    ap.add_argument("--warmup", type=float, default=100.0, help="exclude txs created before this simulated time")
    ap.add_argument("--no-elastic", action="store_true", help="sensitivity run with fixed merge capacity")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="", help="suffix for log dirs / default results file (e.g. _noelastic)")
    ap.add_argument("--results", default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    results = args.results or f"sweep_final_{args.stage}{args.tag}.csv"
    if args.summary:
        summarize(results)
        return

    if args.stage == "tune":
        ns, skews, seeds = [10], [1.0, 2.0], [1, 2, 3, 4, 5]
    else:
        ns, skews, seeds = [4, 6, 10], [0.0, 1.0, 2.0], list(range(6, 16))
    if args.n: ns = parse_list(args.n, int)
    if args.skews: skews = parse_list(args.skews, float)
    if args.seeds: seeds = parse_seeds(args.seeds)

    modes = make_modes(args.stage, args.alpha)
    base = load_base()
    done = already_done(results)
    combos = [(n, s, m, sd) for n in ns for s in skews for m in modes for sd in seeds]
    todo = [c for c in combos if (c[0], c[1], c[2].label, c[3]) not in done]
    print(f"{len(combos)} runs planned, {len(combos) - len(todo)} done, {len(todo)} to run (workers={args.workers})")

    write_header = not os.path.exists(results)
    start = time.time()
    finished = 0
    with open(results, "a", newline="") as f, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(HEADER)
        futs = [pool.submit(run_one, base, args, n, s, m, sd) for (n, s, m, sd) in todo]
        for fut in as_completed(futs):
            (n, s, m, sd), log_path, err = fut.result()
            finished += 1
            if log_path is None:
                print(f"[{finished}/{len(todo)}] FAILED n={n} skew={s} mode={m.label} seed={sd}\n{err}")
                continue
            row = parse_log(log_path)
            writer.writerow([n, s, m.label, sd] + [row[k] for k in METRICS])
            f.flush()
            print(f"[{finished}/{len(todo)}] n={n} skew={s} mode={m.label} seed={sd} "
                  f"avg_lat={row['avg_latency']} (elapsed {time.time() - start:.0f}s)")
    summarize(results)


if __name__ == "__main__":
    main()
