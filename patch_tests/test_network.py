from harness import *
import numpy as np

print("--- unit: staleness semantics ---")
for k in (0, 1, 2):
    params = dict(num_shards=4, verbose=0, info_delay_rounds=k)
    net, env = build(params, 4, 3)
    net._demand_history = netmod.deque(maxlen=k + 1)
    seen = []
    for step in range(5):
        net.get_shard_leader(0).transaction_pool.put(Tx(f"x{step}", 0, 1, 1))      # pending 0->1 grows by 1 each round
        net._record_demand()
        view = net._get_demand_view()[0][1]
        seen.append((step + 1, view))                                              # (true pending, visible pending)
    print(f"k={k}: (true, visible) = {seen}")
    for true, vis in seen:
        assert vis == max(true - k, 1)  # visible = pending as of k rounds ago (early rounds: oldest available)
print("PASS")

print("--- unit: reorder keeps the same pairs, sorts by demand ---")
params = dict(num_shards=6, verbose=0, info_delay_rounds=0)
net, env = build(params, 6, 3); net._demand_history = netmod.deque(maxlen=1)
sched = netmod.algo4_optimal_schedule(6)
for _ in range(7): net.get_shard_leader(2).transaction_pool.put(Tx("a", 0, 1, 5))
for _ in range(3): net.get_shard_leader(0).transaction_pool.put(Tx("b", 0, 1, 1))
net._record_demand(); ordered = net.reorder_rounds_by_weight(sched)
flat = lambda s: sorted(tuple(sorted(p)) for r in s for p in r)
assert flat(ordered) == flat(sched) and len(ordered) == len(sched)
assert (2, 5) in [tuple(sorted(p)) for p in ordered[0]] or (5, 2) in ordered[0]
print("PASS  first round:", ordered[0])

print("--- unit: dynamic gap cap forces a starved pair ---")
params = dict(num_shards=4, verbose=0, info_delay_rounds=0, dynamic_gap_cap_rounds=5, aging_alpha=0.1)
net, env = build(params, 4, 3); net._demand_history = netmod.deque(maxlen=1)
for _ in range(20): net.get_shard_leader(0).transaction_pool.put(Tx("h", 0, 1, 2))      # huge demand 0<->2
net.rounds_since_met = {(i, j): 0 for i in range(4) for j in range(i + 1, 4)}
net._record_demand(); m0 = net._compute_dynamic_round(4, 0.1)
net.rounds_since_met[(0, 1)] = 6; m1 = net._compute_dynamic_round(4, 0.1)
print("no starvation:", m0, "| pair (0,1) starved:", m1)
assert (0, 2) in m0 and (0, 1) in m1
params["dynamic_gap_cap_rounds"] = None; m2 = net._compute_dynamic_round(4, 0.1)
assert (0, 1) not in m2; print("PASS (cap off -> hot pair wins:", m2, ")")

print("--- integration: one run per mode, print -> parse chain ---")
for label, kw in [("vanilla", {}), ("weighted", dict(weighted=True)), ("dynamic", dict(dynamic=True))]:
    net, params, text = run_mode(label, n=6, skew=1.0, seed=3, **kw)
    row = sweep.parse_log_text(text)
    lat = params["accepted_tx_latencies"]
    assert min(lat) >= 1.29, f"latency below one pair duration: {min(lat)}"          # completion-time fix in effect
    assert int(row["residual"]) == 0, row["residual"]
    assert len(lat) == len(params["accepted_tx_pairs"])
    print(f"{label:9s} avg={float(row['avg_latency']):6.2f} p95={float(row['p95_latency']):6.2f} min={min(lat):.2f} "
          f"hot={row['hot_latency']} cold={row['cold_latency']} epoch={row['mean_epoch_duration'] or 'n/a'} "
          f"gap={row['max_gap']} pending={row['pending_end']} rejected={row['rejected']} residual={row['residual']}")
    missing = [k for k, v in row.items() if v == "" and k not in ("epoch", "mean_epoch_duration", "blockchain_length", "cross_included", "accepted_tps", "total_volume", "busiest_pair_pct", "cross_processed", "cross_generated")]
    assert not missing, missing
print("TEST network: PASS")
