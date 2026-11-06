"""
sweep_multi_shard.py

Benchmark Vanilla Sharon vs. Weighted Round Ordering vs. Dynamic Matching
across varying shard counts (n = 3, 4, 6, 8, 10) and traffic skews (0.0 to 2.0).

Usage:
    python sweep_multi_shard.py             # execute full multi-shard sweep
    python sweep_multi_shard.py --summary   # print current aggregated summary table
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
SHARD_COUNTS = [3, 4, 6, 8, 10]
SKEWS = [0.0, 1.0, 1.5, 2.0]
SEEDS = [1, 2, 3]

# MODES: (label, weighted_scheduling, dynamic_matching, aging_alpha)
MODES = [
    ("vanilla",      False, False, None),
    ("weighted",     True,  False, None),
    ("dynamic_a0.1", False, True,  0.1),
]

PARAMS_FILE = "config/params.json"
RESULTS_CSV = "sweep_results_multishard.csv"

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
HEADER = ["num_shards", "traffic_skew", "mode", "seed"] + list(FIELD_PATTERNS.keys())


def load_params():
    with open(PARAMS_FILE, "r") as f:
        return json.load(f)


def save_params(params):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)


def safe_float(val, default=0.0):
    try:
        return float(val) if val != "" else default
    except (ValueError, TypeError):
        return default


def run_one(n_shards, skew, mode, seed):
    label, weighted, dynamic, alpha = mode
    params = load_params()
    params["num_nodes"] = 100
    params["simulation_time"] = 1000
    params["num_shards"] = n_shards
    params["sharon_enabled"] = True
    params["weighted_scheduling"] = weighted
    params["dynamic_matching"] = dynamic
    params["elastic_capacity_enabled"] = True
    if alpha is not None:
        params["aging_alpha"] = alpha
    params["traffic_skew"] = skew
    params["seed"] = seed
    params["verbose"] = 0
    save_params(params)

    dir_suffix = f"sweep_multishard/n{n_shards}_{label}/skew{skew}_seed{seed}"
    result = subprocess.run(["python", "simulate.py", dir_suffix], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] n={n_shards} skew={skew} mode={label} seed={seed}")
        print(result.stdout[-1500:])
        print(result.stderr[-1500:])
        return None

    # Resolve possible log paths
    candidate_dirs = [
        os.path.join("simulation_logs", dir_suffix),
        os.path.join("simulation_logs", "config", "params.json", dir_suffix)
    ]
    log_dir = next((d for d in candidate_dirs if os.path.isdir(d)), None)

    if not log_dir:
        print(f"  [WARN] no log directory found for {dir_suffix}")
        return None

    logs = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    if not logs:
        print(f"  [WARN] no .log file found in {log_dir}")
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
                    done.add((int(r["num_shards"]), float(r["traffic_skew"]), r["mode"], int(r["seed"])))
    return done


def summarize():
    if not os.path.exists(RESULTS_CSV):
        print("No results file yet.")
        return
    with open(RESULTS_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("avg_latency")]

    labels = [m[0] for m in MODES if any(r["mode"] == m[0] for r in rows)]
    
    print("\n=== MULTI-SHARD SHARON BENCHMARK SUMMARY (EACH MODE VS VANILLA) ===")
    print(f"{'n_shards':>8} | {'skew':>5} | {'mode':<12} | {'seeds':>5} | {'vanilla lat':>11} | {'mode lat':>9} | {'mean change':>11} | {'faster in':>9}")
    print("-" * 88)

    for n_shards in sorted({int(r["num_shards"]) for r in rows}):
        for skew in sorted({float(r["traffic_skew"]) for r in rows if int(r["num_shards"]) == n_shards}):
            base = {int(r["seed"]): r for r in rows if int(r["num_shards"]) == n_shards and float(r["traffic_skew"]) == skew and r["mode"] == "vanilla"}
            for label in labels:
                if label == "vanilla":
                    continue
                other = {int(r["seed"]): r for r in rows if int(r["num_shards"]) == n_shards and float(r["traffic_skew"]) == skew and r["mode"] == label}
                seeds = sorted(set(base) & set(other))
                if not seeds:
                    continue
                vl = [safe_float(base[s]["avg_latency"]) for s in seeds]
                ml = [safe_float(other[s]["avg_latency"]) for s in seeds]
                changes = [((b - a) / a * 100) if a > 0 else 0.0 for a, b in zip(vl, ml)]
                wins = sum(1 for c in changes if c < 0)
                
                print(f"{n_shards:8d} | {skew:5.1f} | {label:<12} | {len(seeds):>5} | {statistics.mean(vl):11.3f} | {statistics.mean(ml):9.3f} | {statistics.mean(changes):+10.2f}% | {wins:>4}/{len(seeds):<4}")

    print("\nNote: Negative 'mean change' indicates latency reduction compared to Vanilla Sharon.")


def main():
    if "--summary" in sys.argv:
        summarize()
        return

    done = already_done()
    combos = [(n, s, m, sd) for n in SHARD_COUNTS for s in SKEWS for m in MODES for sd in SEEDS]
    todo = [c for c in combos if (c[0], c[1], c[2][0], c[3]) not in done]
    print(f"{len(combos)} runs planned, {len(combos) - len(todo)} already done, {len(todo)} to run.")

    write_header = not os.path.exists(RESULTS_CSV)
    start = time.time()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(HEADER)
        for i, (n_shards, skew, mode, seed) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] n_shards={n_shards} skew={skew} mode={mode[0]} seed={seed} (elapsed {time.time() - start:.0f}s)")
            log_path = run_one(n_shards, skew, mode, seed)
            if log_path is None:
                continue
            row = parse_log(log_path)
            writer.writerow([n_shards, skew, mode[0], seed] + [row[k] for k in FIELD_PATTERNS])
            f.flush()

    summarize()


if __name__ == "__main__":
    main()
