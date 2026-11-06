"""
sweep_sharon.py (v2)

Compares VANILLA SHARON vs WEIGHTED (MODIFIED) SHARON across a grid of
(num_shards, weighted_scheduling, seed). sharon_enabled is always True -
this sweep is no longer baseline-vs-Sharon, it's Design-1-reordering
vs vanilla, per the project's current focus.

Captures three key metrics per run:
    - cross-shard transactions processed (throughput proxy)
    - average / max cross-shard confirmation latency
    - total accepted cross-shard transaction volume (sum of tx.value)

Usage:
    python sweep_sharon.py

Each run gets its own unique log subfolder (shard count, mode, seed) so
results never overwrite each other.
"""

import json
import os
import re
import subprocess
import time

# ---- Sweep configuration ----
NUM_SHARDS_LIST = [3, 4, 5, 6, 8, 10]
SEEDS = [1, 2, 3]
WEIGHTED_MODES = [False, True]   # vanilla Sharon, then weighted (Design 1) Sharon
PARAMS_FILE = "config/params.json"
RESULTS_CSV = "sweep_results_modified.csv"

FIELD_PATTERNS = {
    "blockchain_length": r"Length of Blockchain = (\d+)",
    "tx_included": r"Total no of transactions included in Blockchain = (\d+)",
    "cross_included": r"Total no of cross-shard transactions included in Blockchain = (\d+)",
    "tx_processed": r"Total no of transactions processed = (\d+)",
    "cross_processed": r"Total no of cross-shard transactions processed = (\d+)",
    "tx_generated": r"Total no of transactions generated = (\d+)",
    "cross_generated": r"Total no of cross-shard transactions generated = (\d+)",
    "processed_tps": r"Processed TPS = ([\d.]+)",
    "accepted_tps": r"Accepted TPS = ([\d.]+)",
    "avg_latency": r"Average cross-shard tx confirmation latency = ([\d.]+)",
    "max_latency": r"Max cross-shard tx confirmation latency = ([\d.]+)",
    "total_volume": r"Total accepted cross-shard tx value \(volume\) = (\d+)",
}


def load_params():
    with open(PARAMS_FILE, "r") as f:
        return json.load(f)


def save_params(params):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)


def run_one(num_shards, weighted_scheduling, seed):
    params = load_params()
    params["num_nodes"] = 100
    params["simulation_time"] = 1000
    params["num_shards"] = num_shards
    params["sharon_enabled"] = True          # always Sharon - comparing vanilla vs weighted
    params["weighted_scheduling"] = weighted_scheduling
    params["seed"] = seed
    params["verbose"] = 0
    save_params(params)

    mode_label = "weighted" if weighted_scheduling else "vanilla"
    dir_suffix = f"sweep_modified/{mode_label}/sh{num_shards}_seed{seed}"

    cmd = ["python", "simulate.py", dir_suffix]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [ERROR] shards={num_shards} weighted={weighted_scheduling} seed={seed}")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return None

    log_dir = os.path.join("simulation_logs", dir_suffix)
    if not os.path.isdir(log_dir):
        print(f"  [WARN] Expected log dir not found: {log_dir}")
        return None

    logs = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    if not logs:
        print(f"  [WARN] No log file produced in {log_dir}")
        return None

    return os.path.join(log_dir, logs[0])


def parse_log(log_path):
    with open(log_path, "r") as f:
        content = f.read()

    row = {}
    for key, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, content)
        row[key] = match.group(1) if match else ""
    return row


def main():
    total_runs = len(NUM_SHARDS_LIST) * len(WEIGHTED_MODES) * len(SEEDS)
    run_idx = 0
    start_time = time.time()

    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a") as csv_f:
        if write_header:
            header = ["num_shards", "weighted_scheduling", "seed"] + list(FIELD_PATTERNS.keys())
            csv_f.write(",".join(header) + "\n")

        for num_shards in NUM_SHARDS_LIST:
            for weighted_scheduling in WEIGHTED_MODES:
                for seed in SEEDS:
                    run_idx += 1
                    elapsed = time.time() - start_time
                    print(
                        f"[{run_idx}/{total_runs}] shards={num_shards} "
                        f"weighted={weighted_scheduling} seed={seed}  "
                        f"(elapsed {elapsed:.0f}s)"
                    )

                    log_path = run_one(num_shards, weighted_scheduling, seed)
                    if log_path is None:
                        continue

                    row = parse_log(log_path)
                    values = [str(num_shards), str(weighted_scheduling), str(seed)] + [
                        row[k] for k in FIELD_PATTERNS.keys()
                    ]
                    csv_f.write(",".join(values) + "\n")
                    csv_f.flush()

    print(f"\nDone. Results written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
