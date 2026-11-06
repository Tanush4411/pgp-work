"""
sweep_sharon.py

Runs simulate.py across a grid of (num_shards, sharon_enabled, seed),
parses each resulting log file, and appends results to a single CSV
for analysis/plotting.

Usage:
    python sweep_sharon.py

Adjust NUM_SHARDS_LIST, SEEDS, and other constants below as needed.
Each run gets its own unique log subfolder (keyed by shard count,
mechanism, and seed) so results never overwrite each other, unlike
plain simulate.py runs within the same minute.
"""

import json
import os
import re
import subprocess
import sys
import time

# ---- Sweep configuration ----
NUM_SHARDS_LIST = [3, 4, 5, 6, 8, 10]   # n=2 excluded: known baseline ShardEval bug (see project notes)
SEEDS = [1, 2, 3]                        # 3 seeds per config for averaging
SHARON_MODES = [False, True]             # baseline, then Sharon
PARAMS_FILE = "config/params.json"
RESULTS_CSV = "sweep_results.csv"

# Fields to extract from each simulation log's SIMULATION DETAILS block
FIELD_PATTERNS = {
    "blockchain_length": r"Length of Blockchain = (\d+)",
    "tx_included": r"Total no of transactions included in Blockchain = (\d+)",
    "intra_included": r"Total no of intra-shard transactions included in Blockchain = (\d+)",
    "cross_included": r"Total no of cross-shard transactions included in Blockchain = (\d+)",
    "tx_processed": r"Total no of transactions processed = (\d+)",
    "intra_processed": r"Total no of intra-shard transactions processed = (\d+)",
    "cross_processed": r"Total no of cross-shard transactions processed = (\d+)",
    "tx_generated": r"Total no of transactions generated = (\d+)",
    "intra_generated": r"Total no of intra-shard transactions generated = (\d+)",
    "cross_generated": r"Total no of cross-shard transactions generated = (\d+)",
    "processed_tps": r"Processed TPS = ([\d.]+)",
    "accepted_tps": r"Accepted TPS = ([\d.]+)",
}


def load_params():
    with open(PARAMS_FILE, "r") as f:
        return json.load(f)


def save_params(params):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)


def run_one(num_shards, sharon_enabled, seed):
    """Configure params.json for this combo, run simulate.py, return path to the produced log."""
    params = load_params()
    params["num_nodes"] = 100   # fixed at 100 (matches original repo's script.py) so shards
                                 # stay large enough for establish_network_connections() to
                                 # find enough neighbours at higher shard counts (n=20 nodes
                                 # was too small - crashes for num_shards >= 5, a pre-existing
                                 # ShardEval baseline limitation unrelated to Sharon).
    params["simulation_time"] = 1000  # reduced from 4000: num_nodes=100 is much slower per
                                       # simulated time-unit than num_nodes=20 was; TPS/accept-rate
                                       # are normalized metrics so this preserves comparability
                                       # while cutting wall-clock runtime roughly 4x.
    params["num_shards"] = num_shards
    params["sharon_enabled"] = sharon_enabled
    params["seed"] = seed
    params["verbose"] = 0
    save_params(params)

    mode_label = "sharon" if sharon_enabled else "baseline"
    dir_suffix = f"sweep/{mode_label}/sh{num_shards}_seed{seed}"

    cmd = ["python", "simulate.py", dir_suffix]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [ERROR] shards={num_shards} sharon={sharon_enabled} seed={seed}")
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
    total_runs = len(NUM_SHARDS_LIST) * len(SHARON_MODES) * len(SEEDS)
    run_idx = 0
    start_time = time.time()

    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a") as csv_f:
        if write_header:
            header = ["num_shards", "sharon_enabled", "seed"] + list(FIELD_PATTERNS.keys())
            csv_f.write(",".join(header) + "\n")

        for num_shards in NUM_SHARDS_LIST:
            for sharon_enabled in SHARON_MODES:
                for seed in SEEDS:
                    run_idx += 1
                    elapsed = time.time() - start_time
                    print(
                        f"[{run_idx}/{total_runs}] shards={num_shards} "
                        f"sharon={sharon_enabled} seed={seed}  "
                        f"(elapsed {elapsed:.0f}s)"
                    )

                    log_path = run_one(num_shards, sharon_enabled, seed)
                    if log_path is None:
                        continue

                    row = parse_log(log_path)
                    values = [str(num_shards), str(sharon_enabled), str(seed)] + [
                        row[k] for k in FIELD_PATTERNS.keys()
                    ]
                    csv_f.write(",".join(values) + "\n")
                    csv_f.flush()

    print(f"\nDone. Results written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
