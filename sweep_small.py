"""
sweep_small.py -- a tiny end-to-end test of the new sweep machinery.

NOT a source of results. It exists to (1) prove that sweep_final.py's pipeline works against your real
simulator, (2) time the runs so you can plan the real sweeps, and (3) run the fairness sanity checks
automatically. Run it from the ShardEval repo root with the venv active:

    python sweep_small.py

It runs 4 modes x 2 seeds = 8 simulations at n=4, skew 1.0 (settings below), never touches
config/params.json, and writes only to sweep_small_results.csv, simulation_logs/sweep_final_small/ and a
temporary sweep_tmp_small/ folder (removed at the end). Anything left over from a previous run of this
script is deleted at the start.

It does NOT start the 630-run main sweep: that is `python sweep_final.py --stage main ...`.
"""

import argparse
import csv
import os
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import sweep_final as sf

# ----------------------------- settings (edit here) -----------------------------
N_SHARDS = 10
SKEW = 2.0
SEEDS = [6, 7, 8, 9, 10]
MODE_LABELS = ["vanilla", "weighted_k0", "dynamic_k0", "dynamic_k1", "dynamic_k2"]
ALPHA = 0.1
NODES = 100
SIM_TIME = 1000          # same as the real sweeps; lower it (e.g. 300) if one run takes too long
WARMUP = None            # None -> automatically tx_start_time + 50
WORKERS = 1              # raise only after you have seen the time per run
TAG = "_n10"
RESULTS = "sweep_n10_results.csv"
TMP = "sweep_tmp_small"
FULL_SWEEP_RUNS = 630    # 3 shard counts x 3 skews x 7 modes x 10 seeds
# --------------------------------------------------------------------------------


def hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m {seconds % 60:02d}s"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    if not os.path.exists(sf.BASE_PARAMS_FILE):
        sys.exit(f"Cannot find {sf.BASE_PARAMS_FILE}. Run this from the ShardEval repo root.")
    if not os.path.exists("simulate.py"):
        sys.exit("Cannot find simulate.py. Run this from the ShardEval repo root.")

    base = sf.load_base()
    tx_start = base.get("tx_start_time")
    cfg_start = base.get("network_config_start_time")
    warmup = WARMUP if WARMUP is not None else (tx_start if tx_start is not None else 0) + 50

    print("=" * 78)
    print("SMALL SWEEP: machinery test, NOT results")
    print("=" * 78)
    print(f"n_shards={N_SHARDS}  skew={SKEW}  seeds={SEEDS}  modes={MODE_LABELS}")
    print(f"nodes={NODES}  simulation_time={SIM_TIME}  workers={WORKERS}")
    print(f"network_config_start_time = {cfg_start}   tx_start_time = {tx_start}")
    print(f"warm-up used = {warmup}  (transactions created before this time are excluded from latency)")
    if tx_start is not None and tx_start > 0:
        print(f"NOTE: no transactions exist before t={tx_start}. Rounds in that period are empty and cost 1 time unit each,")
        print(f"      so the round/epoch diagnostics include about {tx_start} idle time units until diagnostics are trimmed.")
    print()

    # ---- start from a clean slate (only this script's own scratch paths) ----
    for path in (RESULTS,):
        if os.path.exists(path):
            os.remove(path)
    for path in (os.path.join("simulation_logs", f"sweep_final{TAG}"), TMP):
        if os.path.isdir(path):
            shutil.rmtree(path)

    sf.TMP_DIR = TMP
    args = argparse.Namespace(stage="main", tag=TAG, nodes=NODES, sim_time=SIM_TIME, warmup=warmup, no_elastic=False)
    modes = [m for m in sf.make_modes("main", ALPHA) if m.label in MODE_LABELS]
    missing = set(MODE_LABELS) - {m.label for m in modes}
    if missing:
        sys.exit(f"Unknown mode label(s): {sorted(missing)}")
    jobs = [(m, s) for s in SEEDS for m in modes]

    def job(mode, seed):
        t0 = time.time()
        _, log_path, err = sf.run_one(base, args, N_SHARDS, SKEW, mode, seed)
        return mode, seed, log_path, err, time.time() - t0

    rows, durations, failures = [], {}, []
    started = time.time()
    print(f"Running {len(jobs)} simulations...")
    with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
        futs = [pool.submit(job, m, s) for (m, s) in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            mode, seed, log_path, err, dur = fut.result()
            if log_path is None:
                failures.append((mode.label, seed, err))
                print(f"[{i}/{len(jobs)}] FAILED  {mode.label} seed={seed} after {hms(dur)}\n{err}")
                continue
            row = sf.parse_log(log_path)
            row.update(num_shards=N_SHARDS, traffic_skew=SKEW, mode=mode.label, seed=seed)
            rows.append(row)
            durations.setdefault(mode.label, []).append(dur)
            print(f"[{i}/{len(jobs)}] done    {mode.label:<12} seed={seed}  {hms(dur)}  avg_latency={row['avg_latency'] or 'N/A'}")

    with open(RESULTS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(sf.HEADER)
        for r in sorted(rows, key=lambda r: (r["seed"], r["mode"])):
            w.writerow([r["num_shards"], r["traffic_skew"], r["mode"], r["seed"]] + [r[k] for k in sf.METRICS])
    shutil.rmtree(TMP, ignore_errors=True)
    wall = time.time() - started

    # ---------------------------------- per-run table ----------------------------------
    print("\n" + "-" * 78)
    print("PER-RUN VALUES")
    print("-" * 78)
    hdr = f"{'mode':<12} {'seed':>4} {'generated':>9} {'avg lat':>8} {'hot':>7} {'cold':>7} {'p95':>7} {'maxgap':>7} {'pend':>5} {'rej':>4} {'resid':>5} {'tps':>6}"
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["seed"], r["mode"])):
        g = lambda k, fmt="{:.2f}": (fmt.format(fnum(r[k])) if fnum(r[k]) is not None else "n/a")
        print(f"{r['mode']:<12} {r['seed']:>4} {g('cross_generated', '{:.0f}'):>9} {g('avg_latency'):>8} {g('hot_latency'):>7} "
              f"{g('cold_latency'):>7} {g('p95_latency'):>7} {g('max_gap'):>7} {g('pending_end', '{:.0f}'):>5} "
              f"{g('rejected', '{:.0f}'):>4} {g('residual', '{:.0f}'):>5} {g('accepted_tps', '{:.3f}'):>6}")

    # ------------------------------------ checks ------------------------------------
    print("\n" + "-" * 78)
    print("AUTOMATIC CHECKS")
    print("-" * 78)
    problems = []

    def verdict(name, ok, detail="", warn=False):
        tag = "PASS" if ok else ("WARN" if warn else "FAIL")
        print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))
        if not ok and not warn:
            problems.append(name)

    verdict("all simulations completed", not failures and len(rows) == len(jobs),
            f"{len(rows)}/{len(jobs)} runs parsed" + (f"; failed: {[(l, s) for l, s, _ in failures]}" if failures else ""))

    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)

    bad = [(s, sorted({r["cross_generated"] for r in rs})) for s, rs in by_seed.items() if len({r["cross_generated"] for r in rs}) > 1]
    verdict("identical transaction arrivals across modes (cross_generated equal per seed)", not bad,
            f"differs for {bad}" if bad else "equal in every mode for every seed")

    neg = [(r["mode"], r["seed"], r["residual"]) for r in rows if fnum(r["residual"]) is None or fnum(r["residual"]) < 0]
    verdict("conservation residual >= 0 in every run", not neg, f"offending runs: {neg}" if neg else "")

    low = [(r["mode"], r["seed"], r["avg_latency"]) for r in rows if fnum(r["avg_latency"]) is None or fnum(r["avg_latency"]) <= 1.3]
    verdict("average latency above one pair duration (~1.3) in every run", not low, f"offending runs: {low}" if low else "")

    few = [(r["mode"], r["seed"], r["measured_txs"]) for r in rows if fnum(r["measured_txs"]) is None or fnum(r["measured_txs"]) < 100]
    verdict("at least 100 measured transactions per run (after warm-up)", not few,
            f"too few: {few}. Lower the warm-up or raise SIM_TIME" if few else "", warn=True)

    spread = []
    for s, rs in by_seed.items():
        tps = [fnum(r["accepted_tps"]) for r in rs if fnum(r["accepted_tps"])]
        if len(tps) > 1:
            spread.append((s, (max(tps) - min(tps)) / statistics.mean(tps) * 100))
    verdict("throughput (Accepted TPS) within 5% across modes", all(v <= 5 for _, v in spread),
            "spread per seed: " + ", ".join(f"seed {s}: {v:.1f}%" for s, v in spread), warn=True)

    van_epoch = [fnum(r["mean_epoch_duration"]) for r in rows if r["mode"] == "vanilla"]
    verdict("vanilla reports an epoch duration", all(v is not None for v in van_epoch) and bool(van_epoch),
            "new diagnostics not printed? check that simulate.py was replaced" if not all(van_epoch) else "")

    pend_ok = all(fnum(r["pending_end"]) is not None for r in rows)
    verdict("pending-at-end could be measured", pend_ok,
            "" if pend_ok else "send me utils/priority_queue.py", warn=True)

    if tx_start is not None and warmup <= tx_start:
        verdict("warm-up exceeds tx_start_time", False, f"warm-up {warmup} <= tx_start_time {tx_start}: it excludes nothing", warn=True)

    # ------------------------------------- timing -------------------------------------
    print("\n" + "-" * 78)
    print("TIMING")
    print("-" * 78)
    if durations:
        for label in MODE_LABELS:
            if label in durations:
                print(f"  {label:<12} mean {hms(statistics.mean(durations[label]))} per run  ({len(durations[label])} runs)")
        mean_all = statistics.mean([d for ds in durations.values() for d in ds])
        scale = 1000.0 / SIM_TIME
        per_run = mean_all * scale
        note = "" if SIM_TIME == 1000 else f" (scaled x{scale:.1f} from SIM_TIME={SIM_TIME}; approximate)"
        print(f"  wall-clock for this script: {hms(wall)}")
        print(f"  mean per run: {hms(mean_all)}{note}")
        print(f"  projected main sweep ({FULL_SWEEP_RUNS} runs at ~{hms(per_run)} each): "
              f"{hms(per_run * FULL_SWEEP_RUNS)} serial, {hms(per_run * FULL_SWEEP_RUNS / 4)} with 4 workers")
        print("  This used n=4; n=10 runs are probably slower, so treat the projection as a lower bound.")

    # --------------------------------- summary tables ---------------------------------
    print("\n" + "-" * 78)
    print("SUMMARY TABLES (only 2 seeds: confidence intervals are huge, ignore the sizes)")
    print("-" * 78)
    sf.summarize(RESULTS)

    print("\n" + "=" * 78)
    if problems:
        print(f"RESULT: {len(problems)} CHECK(S) FAILED: {problems}")
        print("Send me this whole output. Do not run the real sweeps yet.")
    else:
        print("RESULT: all hard checks passed. The sweep machinery works end to end.")
        print("Delete sweep_small_results.csv and simulation_logs/sweep_final_small/ when you are done looking.")
    print("=" * 78)


if __name__ == "__main__":
    main()
