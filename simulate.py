import os, sys
import math
import random
import simpy
import numpy as np
import json
import pathlib
import time

from network.network import Network
from utils.color_print import ColorPrint


def execute_simulation(name, env, params):
    network_obj = Network(name, env, params)
    network_obj.execute_sybil_resistance_mechanism()

    for idx in range(params["num_epochs"]):
        network_obj.run_epoch()

    """
    To-Do:  Decide where it should be put.
            Putting it before prev loop doesn't generate transactions.
    """
    env.run(until=params["simulation_time"])
    return network_obj


def load_parameters():
    """
    Usage:  python simulate.py [log_dir_suffix] [params_file]
    The params file defaults to config/params.json. (Previously argv[1] -- the log
    directory -- was used as the params file whenever 2+ arguments were given.)
    """
    params_file = sys.argv[2] if len(sys.argv) > 2 else "config/params.json"
    with open(params_file, "r") as f:
        params = json.loads(f.read())
    return params


def percentile(values, q):
    """Linear-interpolated percentile, q in [0, 100]."""
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def print_fairness_metrics(network_obj, params):
    """Extra metrics used by the three-way (Sharon / Design 1 / Design 2) comparison."""
    n = params["num_shards"]
    sim_end = params["simulation_time"]
    lat = params.get("accepted_tx_latencies") or []
    pairs = params.get("accepted_tx_pairs") or []

    # ---- latency distribution -------------------------------------------------
    print(f"\nMeasured cross-shard txs (after warm-up) = {len(lat)}")
    if lat:
        print(f"Average cross-shard tx confirmation latency = {mean(lat):.4f}")
        print(f"Max cross-shard tx confirmation latency = {max(lat):.4f}")
        print(f"P95 cross-shard tx confirmation latency = {percentile(lat, 95):.4f}")
    else:
        print("Average cross-shard tx confirmation latency = N/A")

    # ---- per-pair view: pair-averaged latency, hot vs cold pairs --------------
    by_pair = {}
    for l, p in zip(lat, pairs):
        by_pair.setdefault(tuple(p), []).append(l)
    if by_pair:
        print(f"Pair-averaged cross-shard latency = {mean([mean(v) for v in by_pair.values()]):.4f}")

    tm = params.get("traffic_matrix")
    if tm is not None and by_pair:
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        share = {(i, j): tm[i][j] + tm[j][i] for (i, j) in all_pairs}
        ranked = sorted(all_pairs, key=lambda p: share[p], reverse=True)
        n_hot = max(1, int(math.ceil(0.2 * len(ranked))))
        hot = set(ranked[:n_hot])
        hot_l = [l for p, v in by_pair.items() if p in hot for l in v]
        cold_l = [l for p, v in by_pair.items() if p not in hot for l in v]
        print(f"Hot-pair (top 20% by traffic) avg latency = {mean(hot_l):.4f}")
        print(f"Cold-pair (remaining 80%) avg latency = {mean(cold_l):.4f}")
    else:
        print("Hot-pair (top 20% by traffic) avg latency = N/A")
        print("Cold-pair (remaining 80%) avg latency = N/A")

    # ---- mechanism diagnostics ------------------------------------------------
    rl = params.get("round_log") or []
    if rl:
        print(f"\nNumber of rounds executed = {len(rl)}")
        print(f"Mean round duration = {mean([d for (_, d, _) in rl]):.4f}")
        print(f"Mean active pairs per round = {mean([a for (_, _, a) in rl]):.4f}")
    ed = params.get("epoch_durations") or []
    print(f"Mean epoch duration = {mean(ed):.4f}" if ed else "Mean epoch duration = N/A")

    meets = params.get("pair_meet_times") or {}
    t0 = params.get("tx_start_time", 0)
    per_pair_max = []
    for i in range(n):
        for j in range(i + 1, n):
            ts = sorted(meets.get((i, j), []))
            if not ts:
                per_pair_max.append(sim_end - t0)        # never met: starved for the whole run
                continue
            gaps = [b - a for a, b in zip(ts, ts[1:])] + [sim_end - ts[-1]]
            per_pair_max.append(max(gaps))
    if per_pair_max:
        print(f"Max meeting gap (any pair) = {max(per_pair_max):.4f}")
        print(f"Mean per-pair max meeting gap = {mean(per_pair_max):.4f}")

    # ---- unfinished work + conservation check ---------------------------------
    ages = []
    try:
        for idx in range(n):
            pool = network_obj.get_shard_leader(idx).transaction_pool
            for q in pool.cross_shard_tx_queues.values():
                for tx in q.get(q.length()):
                    ages.append(network_obj.env.now - tx.creation_time)
    except Exception as e:
        ages = None
        print(f"Pending cross-shard txs at end = N/A ({type(e).__name__}: {e})")

    rejected = params.get("rejected_tx_count", 0)
    if ages is not None:
        print(f"Pending cross-shard txs at end = {len(ages)}")
        print(f"Mean age of pending cross-shard txs = {mean(ages):.4f}" if ages else "Mean age of pending cross-shard txs = 0.0000")
        print(f"Max age of pending cross-shard txs = {max(ages):.4f}" if ages else "Max age of pending cross-shard txs = 0.0000")
    print(f"Rejected cross-shard txs = {rejected}")
    print(f"Late completions (after sim end) = {params.get('late_completion_count', 0)}")

    if ages is not None:
        residual = (params["generated_cross_shard_tx_count"]
                    - params["processed_cross_shard_tx_count"] - rejected - len(ages))
        # residual = txs generated but not yet at a leader queue (still in transit). Should be >= 0 and small.
        print(f"Conservation residual (generated - processed - rejected - pending) = {residual}")
        if residual < 0:
            print("WARNING: negative conservation residual -> transactions are being double counted")


def main():
    params = load_parameters()
    seed = params.get("seed", 7)
    np.random.seed(seed)
    random.seed(seed)

    orig_stdout = sys.stdout
    dir_suffix = sys.argv[1] if len(sys.argv) > 1 else time.strftime('%Y-%m-%d/%H-%M')
    dir_name = f"simulation_logs/{dir_suffix}"

    if not os.path.exists(dir_name):
        ColorPrint.print_info(f"\n[Info]: Creating directory '{dir_name}' for storing the simulation logs")
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    file_name = f"{dir_name}/simulation_results_cstx{params['cross_shard_tx_percentage']}_n{params['num_nodes']}_sh{params['num_shards']}_sim{params['simulation_time']}.log"

    f = open(file_name, 'w')
    ColorPrint.print_info(f"\n[Info]: Writing simulation logs to the file '{file_name}'")
    sys.stdout = f

    params["generated_tx_count"], params["processed_tx_count"] = 0, 0
    params["generated_cross_shard_tx_count"], params["generated_intra_shard_tx_count"] = 0, 0
    params["processed_cross_shard_tx_count"], params["processed_intra_shard_tx_count"] = 0, 0

    start_time = time.time()
    env = simpy.Environment()
    network_obj = execute_simulation("Test Network", env, params)
    stop_time = time.time()

    sim_time = stop_time - start_time

    print("\n\n============  SIMULATION DETAILS  ============")
    print(f"\nNumber of nodes = {params['num_nodes']}")
    print(f"Number of shards = {params['num_shards']}")
    print(f"Fraction of cross-shard tx = {params['cross_shard_tx_percentage']}")
    print(f"Simulation Time = {sim_time} seconds")

    if 'chain' in params:
        count, cross_shard_tx_count, intra_shard_tx_count = 0, 0, 0
        for block in params['chain']:
            count += len(block.transactions_list)

            for tx in block.transactions_list:
                intra_shard_tx_count += 1 - tx.cross_shard_status
                cross_shard_tx_count += tx.cross_shard_status

        print(f"\nLength of Blockchain = {len(params['chain'])}")
        print(f"Total no of transactions included in Blockchain = {count}")
        print(f"Total no of intra-shard transactions included in Blockchain = {intra_shard_tx_count}")
        print(f"Total no of cross-shard transactions included in Blockchain = {cross_shard_tx_count}")

        print(f"\nTotal no of transactions processed = {params['processed_tx_count']}")
        print(f"Total no of intra-shard transactions processed = {params['processed_intra_shard_tx_count']}")
        print(f"Total no of cross-shard transactions processed = {params['processed_cross_shard_tx_count']}")

        print(f"\nTotal no of transactions generated = {params['generated_tx_count']}")
        print(f"Total no of intra-shard transactions generated = {params['generated_intra_shard_tx_count']}")
        print(f"Total no of cross-shard transactions generated = {params['generated_cross_shard_tx_count']}")

        print(f"\nProcessed TPS = {params['processed_tx_count']/params['simulation_time']}")
        print(f"Accepted TPS = {count/params['simulation_time']}")

        if params.get("sharon_enabled", False):
            print_fairness_metrics(network_obj, params)
        elif params.get("accepted_tx_latencies"):
            lat = params["accepted_tx_latencies"]
            print(f"\nAverage cross-shard tx confirmation latency = {sum(lat) / len(lat):.4f}")
            print(f"Max cross-shard tx confirmation latency = {max(lat):.4f}")
        else:
            print("\nAverage cross-shard tx confirmation latency = N/A (no accepted tx or Sharon disabled)")

        print(f"Total accepted cross-shard tx value (volume) = {params.get('accepted_tx_value_total', 0)}")
    else:
        print("Simulation didn't execute for sufficiently long time")

    sys.stdout = orig_stdout
    f.close()


if __name__ == "__main__":
    main()
