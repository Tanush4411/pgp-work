"""
sweep_scale.py -- 5 skew levels x 5 shard counts x 3 modes at 5000 nodes, with plots.

Modes compared (all see identical transaction arrivals for a given shard count / skew / seed):
    vanilla   Sharon, oblivious fixed schedule (paper's Algorithms 1-4)
    weighted  Design 1: same schedule, rounds re-ordered each epoch by pending demand
    dynamic   Design 2: a max-weight matching recomputed every round from pending demand (+ aging)
Designs 1 and 2 act on queue counts that are DELAY rounds old (0 = free, instantaneous information).

Put this file next to sweep_final.py in the ShardEval repo root. Run from a terminal with the venv active.

    python sweep_scale.py                       # prints the plan and exits (starts nothing)
    python sweep_scale.py --pilot               # times 2 short runs at 5000 nodes, projects the total time
    python sweep_scale.py --run --workers 1     # runs the sweep (resumable: finished runs are skipped)
    python sweep_scale.py --plots               # (re)draws the plots from whatever has finished so far

5000 nodes is ~50x the 100-node runs you have done so far (~1 min each). Run --pilot first.
Results go to sweep_scale_results.csv, plots to sweep_scale_plots/. config/params.json is never modified.
"""

import argparse
import csv
import math
import os
import resource
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")                      # no display needed (WSL / servers)
import matplotlib.pyplot as plt
import numpy as np

import sweep_final as sf

# ----------------------------- defaults (edit here or use flags) -----------------------------
NODES = 5000
SIM_TIME = 1000
SHARD_COUNTS = [10, 20, 30, 40, 50]        # at most 500 .. 100 nodes per shard (the paper uses ~100-node shards)
SKEWS = [0.0, 0.5, 1.0, 1.5, 2.0]
SEEDS = [6]                                # 1 seed = 75 runs. More seeds -> error bars, but proportionally more time
ALPHA = 0.1                                # Design 2 aging term
DELAY = 0                                  # information delay in rounds for Designs 1 and 2
WARMUP = 100.0                             # transactions created before this time are excluded from latency
RESULTS = "sweep_scale_results.csv"
PLOT_DIR = "sweep_scale_plots"
TAG = "_scale"
TMP = "sweep_tmp_scale"
# ---------------------------------------------------------------------------------------------

sf.METRICS.setdefault("total_processed", r"Total no of transactions processed = (\S+)")
HEADER = ["num_shards", "traffic_skew", "mode", "seed"] + list(sf.METRICS.keys())

MODE_STYLE = {
    "vanilla": ("Sharon (vanilla)", "#555555", "o"),
    "weighted": ("Design 1: weighted order", "#1f77b4", "s"),
    "dynamic": ("Design 2: dynamic matching", "#d62728", "^"),
}


def hms(seconds):
    seconds = int(round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def fl(x):
    try:
        return float(x) if x not in ("", None) else None
    except ValueError:
        return None


def make_modes(delay, alpha):
    return [
        sf.Mode("vanilla", False, False, None, 0),
        sf.Mode("weighted", True, False, None, delay),
        sf.Mode("dynamic", False, True, alpha, delay),
    ]


def make_args(a, sim_time=None, warmup=None, tag=None):
    return argparse.Namespace(stage="scale", tag=tag if tag is not None else TAG, nodes=a.nodes,
                              sim_time=sim_time if sim_time is not None else a.sim_time,
                              warmup=warmup if warmup is not None else a.warmup, no_elastic=a.no_elastic)


def parse_list(s, cast):
    return [cast(x) for x in s.split(",") if x.strip() != ""]


# ------------------------------------------ plan ------------------------------------------
def print_plan(a, ns, skews, seeds):
    runs = len(ns) * len(skews) * len(seeds) * 3
    print("=" * 78)
    print("SCALE SWEEP PLAN (nothing has been started)")
    print("=" * 78)
    print(f"nodes={a.nodes}  simulation_time={a.sim_time}  warm-up={a.warmup}")
    print(f"shard counts = {ns}   (at most {a.nodes // max(ns)} to {a.nodes // min(ns)} nodes per shard)")
    print(f"skews        = {skews}")
    print(f"seeds        = {seeds}")
    print(f"modes        = vanilla, weighted (delay={a.delay} rounds), dynamic (delay={a.delay} rounds, alpha={a.alpha})")
    print(f"total runs   = {len(ns)} x {len(skews)} x 3 x {len(seeds)} = {runs}")
    print()
    print("Cost warning: your 100-node runs took about 50 s. With 5000 nodes there are ~50x more transactions")
    print("(roughly 100,000 cross-shard transactions per run) and hundreds of voters per joint shard, so one run may take")
    print("many minutes to hours, and each run may need gigabytes of memory. Measure it first:")
    print()
    print("    python sweep_scale.py --pilot")
    print()
    print("then start with:   python sweep_scale.py --run --workers 1")
    print("To shrink the job:  --n 10,30,50   --skews 0,1,2   --sim-time 500   --nodes 2000   (all flags override the defaults)")
    print("=" * 78)


# ------------------------------------------ pilot ------------------------------------------
def pilot(a, ns, skews, seeds):
    base = sf.load_base()
    sf.TMP_DIR = TMP
    n_mid = sorted(ns)[len(ns) // 2]
    mode = make_modes(a.delay, a.alpha)[0]
    t_short, t_long = 50, 100
    print(f"PILOT: vanilla, n={n_mid}, skew=1.0, nodes={a.nodes}, two short runs (sim_time {t_short} and {t_long}).")
    print("This separates fixed set-up cost from the cost per simulated time unit. It can take several minutes.\n")
    times = {}
    for T in (t_short, t_long):
        args = make_args(a, sim_time=T, warmup=0.0, tag=TAG + "_pilot")
        t0 = time.time()
        _, log_path, err = sf.run_one(base, args, n_mid, 1.0, mode, seeds[0])
        dur = time.time() - t0
        if log_path is None:
            print(f"Pilot run (sim_time={T}) FAILED after {hms(dur)}:\n{err}")
            return
        times[T] = dur
        print(f"  sim_time={T:4d}: {hms(dur)}")
    peak_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6      # KB -> GB on Linux
    b = (times[t_long] - times[t_short]) / (t_long - t_short)
    c = times[t_short] - b * t_short
    runs = len(ns) * len(skews) * len(seeds) * 3
    print(f"\n  peak memory of one run ~ {peak_gb:.1f} GB")
    if b > 0.005:
        per_runs = [("estimate", c + b * a.sim_time)]
        print(f"  fixed set-up cost ~ {hms(max(c, 0))}, cost per simulated time unit ~ {b:.2f} s")
    else:
        # the two timings do not separate set-up cost from per-time-unit cost: report both extremes
        lo, hi = times[t_long], times[t_long] * a.sim_time / t_long
        per_runs = [("lower bound (all cost is fixed set-up)", lo), ("upper bound (all cost scales with sim time)", hi)]
        print("  the two pilot timings are too close to separate set-up cost from per-time-unit cost,")
        print("  so the projection below is a range.")
    print(f"\nProjection for {runs} runs at sim_time={a.sim_time} (this cell only; the others will differ):")
    for label, per_run in per_runs:
        print(f"  {label}: per run ~ {hms(per_run)}")
        for w in (1, 2, 4):
            print(f"      {w} worker(s): ~ {hms(per_run * runs / w)}   (needs ~{peak_gb * w:.0f} GB of memory)")
    print("\nSmaller n means bigger shards and more voters per transaction, so n=10 runs will be slower than this and")
    print("n=50 runs faster; dynamic at n=50 also pays for a larger matching each round. Treat this as +-2x.")


# ------------------------------------------- run -------------------------------------------
def already_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((int(r["num_shards"]), float(r["traffic_skew"]), r["mode"], int(r["seed"])))
    return done


def run_sweep(a, ns, skews, seeds):
    base = sf.load_base()
    sf.TMP_DIR = TMP
    modes = make_modes(a.delay, a.alpha)
    args = make_args(a)
    done = already_done(a.results)
    jobs = [(n, s, m, sd) for sd in seeds for n in ns for s in skews for m in modes]      # complete one seed's grid first
    todo = [j for j in jobs if (j[0], j[1], j[2].label, j[3]) not in done]
    print(f"{len(jobs)} runs planned, {len(jobs) - len(todo)} already done, {len(todo)} to run (workers={a.workers})")
    if not todo:
        return
    write_header = not os.path.exists(a.results)
    pool = ThreadPoolExecutor(max_workers=max(1, a.workers))
    futs = [pool.submit(sf.run_one, base, args, n, s, m, sd) for (n, s, m, sd) in todo]
    started, finished = time.time(), 0
    try:
        with open(a.results, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(HEADER)
            for fut in as_completed(futs):
                (n, s, m, sd), log_path, err = fut.result()
                finished += 1
                if log_path is None:
                    print(f"[{finished}/{len(todo)}] FAILED n={n} skew={s} mode={m.label} seed={sd}\n{err}")
                    continue
                row = sf.parse_log(log_path)
                writer.writerow([n, s, m.label, sd] + [row.get(k, "") for k in sf.METRICS])
                f.flush()
                elapsed = time.time() - started
                eta = elapsed / finished * (len(todo) - finished)
                print(f"[{finished}/{len(todo)}] n={n} skew={s} mode={m.label} seed={sd} "
                      f"avg_lat={row['avg_latency'] or 'N/A'} processed={row['cross_processed']}  "
                      f"(elapsed {hms(elapsed)}, ~{hms(eta)} left)")
    except KeyboardInterrupt:
        for fut in futs:
            fut.cancel()
        pool.shutdown(wait=False)
        print("\nStopped. Finished runs are saved; re-run with --run to resume. (Running simulations were interrupted.)")
        return
    pool.shutdown(wait=True)
    shutil_rm(TMP)


def shutil_rm(path):
    shutil.rmtree(path, ignore_errors=True)


# ------------------------------------------- plots -------------------------------------------
def cell(rows, mode, key, n, skew):
    vals = [fl(r.get(key)) for r in rows
            if int(r["num_shards"]) == n and float(r["traffic_skew"]) == skew and r["mode"] == mode]
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), 0.0
    m, h = sf.ci95(vals)
    return m, (0.0 if math.isnan(h) else h)


def panel_figure(rows, key, ylabel, title, fname, outdir, ns, skews, show_generated=False, logy=False):
    fig, axes = plt.subplots(1, len(ns), figsize=(3.7 * len(ns), 4.2), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], ns):
        for mode, (label, color, marker) in MODE_STYLE.items():
            ys, es = zip(*[cell(rows, mode, key, n, s) for s in skews])
            ax.errorbar(skews, ys, yerr=es, label=label, color=color, marker=marker, capsize=3, lw=1.6, ms=5)
        if show_generated:
            ys = [cell(rows, "vanilla", "cross_generated", n, s)[0] for s in skews]
            ax.plot(skews, ys, "k--", lw=1, label="generated (all modes)")
        ax.set_title(f"{n} shards")
        ax.set_xlabel("traffic skew")
        ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
    axes[0][0].set_ylabel(ylabel)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def two_row_figure(rows, keys, ylabels, title, fname, outdir, ns, skews):
    fig, axes = plt.subplots(2, len(ns), figsize=(3.7 * len(ns), 7), sharey="row", squeeze=False)
    for r, (key, ylabel) in enumerate(zip(keys, ylabels)):
        for ax, n in zip(axes[r], ns):
            for mode, (label, color, marker) in MODE_STYLE.items():
                ys, es = zip(*[cell(rows, mode, key, n, s) for s in skews])
                ax.errorbar(skews, ys, yerr=es, label=label, color=color, marker=marker, capsize=3, lw=1.6, ms=5)
            ax.set_title(f"{n} shards")
            ax.grid(alpha=0.3)
            if r == 1:
                ax.set_xlabel("traffic skew")
        axes[r][0].set_ylabel(ylabel)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def heatmap_figure(rows, key, title, fname, outdir, ns, skews):
    modes = ["weighted", "dynamic"]
    mats = {}
    for mode in modes:
        M = np.full((len(ns), len(skews)), np.nan)
        for i, n in enumerate(ns):
            for j, s in enumerate(skews):
                v = sf.paired_pct(rows, "vanilla", mode, key, n, s)
                if v:
                    M[i, j] = float(np.mean(v))
        mats[mode] = M
    finite = np.concatenate([m[np.isfinite(m)] for m in mats.values()]) if any(np.isfinite(m).any() for m in mats.values()) else np.array([10.0])
    lim = max(10.0, float(np.max(np.abs(finite))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        M = mats[mode]
        im = ax.imshow(M, cmap="RdYlGn_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(skews)))
        ax.set_xticklabels([str(s) for s in skews])
        ax.set_yticks(range(len(ns)))
        ax.set_yticklabels([str(n) for n in ns])
        ax.set_xlabel("traffic skew")
        ax.set_ylabel("number of shards")
        ax.set_title(MODE_STYLE[mode][0])
        for i in range(len(ns)):
            for j in range(len(skews)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.0f}%", ha="center", va="center", fontsize=9,
                            color="white" if abs(M[i, j]) > 0.55 * lim else "black")
        fig.colorbar(im, ax=ax, label="% change vs vanilla (green = lower latency)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)
    return mats


def make_plots(a, ns_arg, skews_arg):
    if not os.path.exists(a.results):
        print(f"No results file {a.results} yet.")
        return
    with open(a.results, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Results file is empty.")
        return
    ns = sorted({int(r["num_shards"]) for r in rows})
    skews = sorted({float(r["traffic_skew"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    os.makedirs(a.plot_dir, exist_ok=True)
    sub = f"{a.nodes} nodes, {a.sim_time} time units, info delay {a.delay} round(s); {len(seeds)} seed(s)" + \
          ("; bars = 95% CI" if len(seeds) > 1 else "")
    P = lambda *args, **kw: panel_figure(rows, *args, outdir=a.plot_dir, ns=ns, skews=skews, **kw)

    P("cross_processed", "cross-shard transactions processed", f"Transactions processed vs traffic skew ({sub})",
      "fig1_transactions_processed.png", show_generated=True)
    P("avg_latency", "average confirmation latency (time units)", f"Average latency vs traffic skew ({sub})",
      "fig2_avg_latency.png")
    mats = heatmap_figure(rows, "avg_latency",
                          f"Change in average latency vs vanilla ({a.nodes} nodes, {a.sim_time} time units, "
                          f"info delay {a.delay} round(s); {len(seeds)} seed(s))",
                          "fig3_latency_change_heatmap.png", a.plot_dir, ns, skews)
    P("p95_latency", "95th-percentile latency (time units)", f"Tail latency (p95) vs traffic skew ({sub})",
      "fig4_p95_latency.png")
    two_row_figure(rows, ["hot_latency", "cold_latency"],
                   ["hot pairs: avg latency", "cold pairs: avg latency"],
                   f"Who benefits? Hot (top 20% by traffic) vs cold pairs ({sub})", "fig5_hot_vs_cold.png",
                   a.plot_dir, ns, skews)
    P("max_gap", "longest gap between a pair's meetings", f"Worst-case wait between meetings (starvation) ({sub})",
      "fig6_max_meeting_gap.png")
    P("pending_end", "cross-shard txs still pending at end", f"Unfinished work at the end of the run ({sub})",
      "fig7_pending_at_end.png")
    P("rejected", "cross-shard txs rejected by vote", f"Rejected transactions (shard size effect) ({sub})",
      "fig8_rejected.png")

    print(f"\nPlots written to {a.plot_dir}/ (8 figures).")
    print("\nChange in average latency vs vanilla (paired by seed; negative = better):")
    for mode in ("weighted", "dynamic"):
        print(f"\n  {MODE_STYLE[mode][0]}      " + "".join(f"skew {s:<5}" for s in skews))
        for i, n in enumerate(ns):
            print(f"  n={n:<3d}                        " + "".join(
                f"{mats[mode][i, j]:+7.1f}%  " if np.isfinite(mats[mode][i, j]) else "     n/a  " for j in range(len(skews))))


# ------------------------------------------- main -------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="run the sweep (resumable)")
    ap.add_argument("--pilot", action="store_true", help="time two short runs and project the total time")
    ap.add_argument("--plots", action="store_true", help="draw plots from the results collected so far")
    ap.add_argument("--nodes", type=int, default=NODES)
    ap.add_argument("--sim-time", type=int, default=SIM_TIME)
    ap.add_argument("--n", default=",".join(map(str, SHARD_COUNTS)), help="comma list of shard counts")
    ap.add_argument("--skews", default=",".join(map(str, SKEWS)), help="comma list of traffic skews")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="comma list or range like 6-8")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--delay", type=int, default=DELAY, help="information delay in rounds (Designs 1 and 2)")
    ap.add_argument("--warmup", type=float, default=WARMUP)
    ap.add_argument("--no-elastic", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--plot-dir", default=PLOT_DIR)
    a = ap.parse_args()

    ns, skews = parse_list(a.n, int), parse_list(a.skews, float)
    seeds = sf.parse_seeds(a.seeds)
    if not os.path.exists("simulate.py") or not os.path.exists(sf.BASE_PARAMS_FILE):
        sys.exit("Run this from the ShardEval repo root (simulate.py and config/params.json must exist).")

    if a.pilot:
        pilot(a, ns, skews, seeds)
    elif a.run:
        run_sweep(a, ns, skews, seeds)
        make_plots(a, ns, skews)
    elif a.plots:
        make_plots(a, ns, skews)
    else:
        print_plan(a, ns, skews, seeds)


if __name__ == "__main__":
    main()
