import csv, time, sys
from harness import *
modes = [("vanilla", {})] + \
        [(f"weighted_k{k}", dict(weighted=True, k=k)) for k in (0, 1, 2)] + \
        [(f"dynamic_k{k}", dict(dynamic=True, alpha=0.1, k=k)) for k in (0, 1, 2)]
rows, t0 = [], time.time()
for n in (4, 10):
    for skew in (0.0, 2.0):
        for seed in range(1, 11):
            for label, kw in modes:
                net, params, text = run_mode(label, n, skew, seed, **kw)
                r = sweep.parse_log_text(text)
                rows.append([n, skew, label, seed] + [r[k] for k in sweep.METRICS])
with open("toy_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(sweep.HEADER); w.writerows(rows)
print(f"{len(rows)} toy runs in {time.time()-t0:.0f}s")
