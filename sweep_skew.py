"""
sweep_skew.py

Vanilla Sharon vs weighted-order Sharon (Design 1) under increasing
cross-shard traffic skew, at a fixed shard count.

Same seed => same hot shard pairs in both modes, so results are compared
PAIRED, per seed.

Usage:
    python sweep_skew.py             # run the sweep (skips runs already in the CSV)
    python sweep_skew.py --summary   # just print the summary from the CSV

If the sweep is interrupted, run it again: finished runs are not repeated.
Delete sweep_results_skew.csv to start from scratch.
"""

import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time

# ---- Sweep configuration ----
NUM_SHARDS = 10
SKEWS = [0.0, 1.0, 1.5, 2.0]
SEEDS = [1, 2, 3, 4, 5]
WEIGHTED_MODES = [False, True]      # vanilla first, then weighted
PARAMS_FILE = "config/params.json"
RESULTS_CSV = "sweep_results_skew.csv"

FIELD_PATTERNS = {
    "blockchain_length": r"Length of Blockchain = (\d+)",
    "cross_included": r"Total no of cross-shard transactions included in Blockchain = (\d+)",
    "cross_processed": r"Total no of cross-shard transactions processed = (\d+)",
    "cross_generated": r"Total no of cross-shard transactions generated = (\d+)",
    "accepted_tps": r"Accepted TPS = ([\d.]+)",
    "avg_latency": r"Average cross-shard tx confirmation latency = ([\d.]+)",
    "max_latency": r"Max cross-shard tx confirmation latency = ([\d.]+)",
    "total_volume": r"Total accepted cross-shard tx value \(volume\) = (\d+)",
    "busiest_pair_pct": r"busiest pair carries ([\d.]+)%",
}
HEADER = ["traffic_skew", "weighted_scheduling", "seed"] + list(FIELD_PATTERNS.keys())


def load_params():
    with open(PARAMS_FILE, "r") as f:
        return json.load(f)


def save_params(params):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)


def run_one(skew, weighted, seed):
    params = load_params()
    params["num_nodes"] = 100
    params["simulation_time"] = 1000
    params["num_shards"] = NUM_SHARDS
    params["sharon_enabled"] = True
    params["dynamic_matching"] = False          # forced off: it would override weighted_scheduling
    params["weighted_scheduling"] = weighted
    params["traffic_skew"] = skew
    params["seed"] = seed
    params["verbose"] = 0
    save_params(params)

    mode = "weighted" if weighted else "vanilla"
    dir_suffix = f"sweep_skew/{mode}/skew{skew}_seed{seed}"

    result = subprocess.run(["python", "simulate.py", dir_suffix], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] skew={skew} weighted={weighted} seed={seed}")
        print(result.stdout[-1500:])
        print(result.stderr[-1500:])
        return None

    log_dir = os.path.join("simulation_logs", dir_suffix)
    logs = [f for f in os.listdir(log_dir) if f.endswith(".log")] if os.path.isdir(log_dir) else []
    if not logs:
        print(f"  [WARN] no log found in {log_dir}")
        return None
    return os.path.join(log_dir, logs[0])


def parse_log(path):
    with open(path, "r") as f:
        content = f.read()
    row = {}
    for key, pattern in FIELD_PATTERNS.items():
        m = re.search(pattern, content)
        row[key] = m.group(1) if m else ""
    return row


def already_done():
    """Set of (skew, weighted, seed) already present in the CSV."""
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("avg_latency"):
                    done.add((float(r["traffic_skew"]), r["weighted_scheduling"] == "True", int(r["seed"])))
    return done


def summarize():
    if not os.path.exists(RESULTS_CSV):
        print("No results file yet.")
        return
    with open(RESULTS_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("avg_latency")]

    print(f"\n=== Vanilla vs weighted-order Sharon, n={NUM_SHARDS} shards (paired by seed) ===")
    print(f"{'skew':>5} | {'busiest pair':>12} | {'vanilla lat':>11} | {'weighted lat':>12} | "
          f"{'mean change':>11} | {'weighted lower':>14} | {'processed share v/w':>19}")
    for skew in sorted({float(r["traffic_skew"]) for r in rows}):
        v = {int(r["seed"]): r for r in rows if float(r["traffic_skew"]) == skew and r["weighted_scheduling"] == "False"}
        w = {int(r["seed"]): r for r in rows if float(r["traffic_skew"]) == skew and r["weighted_scheduling"] == "True"}
        seeds = sorted(set(v) & set(w))
        if not seeds:
            continue
        vl = [float(v[s]["avg_latency"]) for s in seeds]
        wl = [float(w[s]["avg_latency"]) for s in seeds]
        changes = [(b - a) / a * 100 for a, b in zip(vl, wl)]
        wins = sum(1 for c in changes if c < 0)
        share_v = statistics.mean(float(v[s]["cross_processed"]) / float(v[s]["cross_generated"]) for s in seeds) * 100
        share_w = statistics.mean(float(w[s]["cross_processed"]) / float(w[s]["cross_generated"]) for s in seeds) * 100
        busiest = [float(v[s]["busiest_pair_pct"]) for s in seeds if v[s]["busiest_pair_pct"]]
        busiest_txt = f"{statistics.mean(busiest):.1f}%" if busiest else "uniform"
        print(f"{skew:5.1f} | {busiest_txt:>12} | {statistics.mean(vl):11.3f} | {statistics.mean(wl):12.3f} | "
              f"{statistics.mean(changes):+10.2f}% | {wins:>6}/{len(seeds):<7} | {share_v:8.1f}% /{share_w:6.1f}%")
    print("\nlatency is in simulated time units; 'mean change' < 0 means weighted is faster.")
    print("'processed share' = cross-shard processed / generated; a drop at high skew means hot pairs saturate.")


def main():
    if "--summary" in sys.argv:
        summarize()
        return

    done = already_done()
    combos = [(s, w, sd) for s in SKEWS for w in WEIGHTED_MODES for sd in SEEDS]
    todo = [c for c in combos if c not in done]
    print(f"{len(combos)} runs planned, {len(combos) - len(todo)} already done, {len(todo)} to run.")

    write_header = not os.path.exists(RESULTS_CSV)
    start = time.time()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(HEADER)
        for i, (skew, weighted, seed) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] skew={skew} weighted={weighted} seed={seed}  (elapsed {time.time() - start:.0f}s)")
            log_path = run_one(skew, weighted, seed)
            if log_path is None:
                continue
            row = parse_log(log_path)
            writer.writerow([skew, weighted, seed] + [row[k] for k in FIELD_PATTERNS])
            f.flush()

    summarize()


if __name__ == "__main__":
    main()
