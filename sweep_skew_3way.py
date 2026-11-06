"""
sweep_skew_3way.py

Vanilla Sharon vs weighted-order Sharon vs dynamic-matching Sharon, under
increasing cross-shard traffic skew, at a fixed shard count.

Same seed => same hot shard pairs in every mode, so each mode is compared
to vanilla PAIRED, per seed.

Usage:
    python sweep_skew_3way.py             # run the sweep (finished runs are skipped)
    python sweep_skew_3way.py --summary   # just print the summary from the CSV

If interrupted, run it again. The CSV from the earlier two-mode sweep
(with a weighted_scheduling column) is converted in place; a copy is kept
as <csv>.bak, and its finished runs are not repeated.
"""

import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

# ---- Sweep configuration ----
NUM_SHARDS = 10
SKEWS = [0.0, 1.0, 1.5, 2.0]
SEEDS = [1, 2, 3, 4, 5]
# (label, weighted_scheduling, dynamic_matching, aging_alpha)
MODES = [
    ("vanilla",    False, False, None),
    ("weighted",   True,  False, None),
    ("dynamic_a5", False, True,  5.0),
]
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
HEADER = ["traffic_skew", "mode", "seed"] + list(FIELD_PATTERNS.keys())


def load_params():
    with open(PARAMS_FILE, "r") as f:
        return json.load(f)


def save_params(params):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)


def migrate_old_csv():
    """Convert a v1 CSV (weighted_scheduling True/False) to the 'mode' format."""
    if not os.path.exists(RESULTS_CSV):
        return
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    if "mode" in fields or "weighted_scheduling" not in fields:
        return
    shutil.copy(RESULTS_CSV, RESULTS_CSV + ".bak")
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for r in rows:
            mode = "weighted" if r["weighted_scheduling"] == "True" else "vanilla"
            writer.writerow([r["traffic_skew"], mode, r["seed"]] + [r.get(k, "") for k in FIELD_PATTERNS])
    print(f"Converted {len(rows)} old rows to the new format (backup: {RESULTS_CSV}.bak)")


def run_one(skew, mode, seed):
    label, weighted, dynamic, alpha = mode
    params = load_params()
    params["num_nodes"] = 100
    params["simulation_time"] = 1000
    params["num_shards"] = NUM_SHARDS
    params["sharon_enabled"] = True
    params["weighted_scheduling"] = weighted
    params["dynamic_matching"] = dynamic
    if alpha is not None:
        params["aging_alpha"] = alpha
    params["traffic_skew"] = skew
    params["seed"] = seed
    params["verbose"] = 0
    save_params(params)

    dir_suffix = f"sweep_skew/{label}/skew{skew}_seed{seed}"
    result = subprocess.run(["python", "simulate.py", dir_suffix], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] skew={skew} mode={label} seed={seed}")
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
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("avg_latency"):
                    done.add((float(r["traffic_skew"]), r["mode"], int(r["seed"])))
    return done


def summarize():
    if not os.path.exists(RESULTS_CSV):
        print("No results file yet.")
        return
    with open(RESULTS_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("avg_latency")]

    def pick(skew, label):
        return {int(r["seed"]): r for r in rows
                if float(r["traffic_skew"]) == skew and r["mode"] == label}

    labels = [m[0] for m in MODES if any(r["mode"] == m[0] for r in rows)]
    print(f"\n=== n={NUM_SHARDS} shards, each mode vs vanilla, paired by seed ===")
    print(f"{'skew':>5} | {'mode':<10} | {'seeds':>5} | {'vanilla lat':>11} | {'mode lat':>9} | "
          f"{'mean change':>11} | {'faster in':>9} | {'max lat v -> m':>15} | {'processed v -> m':>17}")
    for skew in sorted({float(r["traffic_skew"]) for r in rows}):
        base = pick(skew, "vanilla")
        for label in labels:
            if label == "vanilla":
                continue
            other = pick(skew, label)
            seeds = sorted(set(base) & set(other))
            if not seeds:
                continue
            vl = [float(base[s]["avg_latency"]) for s in seeds]
            ml = [float(other[s]["avg_latency"]) for s in seeds]
            changes = [(b - a) / a * 100 for a, b in zip(vl, ml)]
            wins = sum(1 for c in changes if c < 0)
            vmax = statistics.mean(float(base[s]["max_latency"]) for s in seeds)
            mmax = statistics.mean(float(other[s]["max_latency"]) for s in seeds)
            vshare = statistics.mean(float(base[s]["cross_processed"]) / float(base[s]["cross_generated"]) for s in seeds) * 100
            mshare = statistics.mean(float(other[s]["cross_processed"]) / float(other[s]["cross_generated"]) for s in seeds) * 100
            print(f"{skew:5.1f} | {label:<10} | {len(seeds):>5} | {statistics.mean(vl):11.3f} | {statistics.mean(ml):9.3f} | "
                  f"{statistics.mean(changes):+10.2f}% | {wins:>4}/{len(seeds):<4} | "
                  f"{vmax:6.1f} ->{mmax:6.1f} | {vshare:7.1f}% ->{mshare:6.1f}%")
    print("\nlatency is in simulated time units; 'mean change' < 0 means faster than vanilla.")
    print("max latency rising = some pairs wait longer (starvation); processed share dropping = backlog.")
    print("Only compare latency within one skew level, never across levels.")


def main():
    migrate_old_csv()
    if "--summary" in sys.argv:
        summarize()
        return

    done = already_done()
    combos = [(s, m, sd) for s in SKEWS for m in MODES for sd in SEEDS]
    todo = [c for c in combos if (c[0], c[1][0], c[2]) not in done]
    print(f"{len(combos)} runs planned, {len(combos) - len(todo)} already done, {len(todo)} to run.")

    write_header = not os.path.exists(RESULTS_CSV)
    start = time.time()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(HEADER)
        for i, (skew, mode, seed) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] skew={skew} mode={mode[0]} seed={seed}  (elapsed {time.time() - start:.0f}s)")
            log_path = run_one(skew, mode, seed)
            if log_path is None:
                continue
            row = parse_log(log_path)
            writer.writerow([skew, mode[0], seed] + [row[k] for k in FIELD_PATTERNS])
            f.flush()

    summarize()


if __name__ == "__main__":
    main()
