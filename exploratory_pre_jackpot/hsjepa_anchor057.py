"""Compare the 3 kbs56774 anchor versions (= ~0.5677 best line) + inspect distributions.
Goal: find a transfer-safe improvement we can make WITHOUT the missing jackpot infra."""
import numpy as np, pandas as pd
from pathlib import Path
RAW = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\outputs\raw")
DL = Path(r"C:\Users\박광민\Downloads")
TARGETS = ["Q1","Q2","Q3","S1","S2","S3","S4"]
KEYS = ["subject_id","sleep_date","lifelog_date"]

files = {"v0_0603": DL/"kbs56774.csv", "v1_0606": DL/"kbs56774 (1).csv", "v2_0610": DL/"kbs56774 (2).csv"}
dfs = {}
for k,p in files.items():
    d = pd.read_csv(p)
    d = d.sort_values(KEYS).reset_index(drop=True)
    dfs[k] = d
    print(f"{k}: shape {d.shape} range[{d[TARGETS].min().min():.4f},{d[TARGETS].max().max():.4f}] "
          f"means " + " ".join(f"{t}={d[t].mean():.3f}" for t in TARGETS))

print("\n=== pairwise mean|diff| between versions ===")
ks = list(dfs)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        a,b = dfs[ks[i]],dfs[ks[j]]
        d = np.abs(a[TARGETS].values-b[TARGETS].values)
        print(f"  {ks[i]} vs {ks[j]}: mean|d|={d.mean():.5f} max|d|={d.max():.4f} changed_cells={(d>1e-9).sum()}")

print("\n=== newest (v2_0610) prediction distribution per target (overconfidence?) ===")
v2 = dfs["v2_0610"]
for t in TARGETS:
    p = v2[t].values
    print(f"  {t}: <0.05:{(p<0.05).mean()*100:4.1f}%  <0.1:{(p<0.1).mean()*100:4.1f}%  "
          f">0.9:{(p>0.9).mean()*100:4.1f}%  >0.95:{(p>0.95).mean()*100:4.1f}%  "
          f"min={p.min():.4f} max={p.max():.4f}")

# train label marginals for reference
lbl = pd.read_csv(RAW/"ch2026_metrics_train.csv")
print("\n  train marginals:", {t:round(lbl[t].mean(),3) for t in TARGETS})
print("  v2 means       :", {t:round(v2[t].mean(),3) for t in TARGETS})

# sample key check
sample = pd.read_csv(RAW/"ch2026_submission_sample.csv")
print("\n  v2 keys == sample keys:", v2[KEYS].reset_index(drop=True).equals(sample[KEYS].reset_index(drop=True)))
print("  v2 shape == sample:", v2.shape==sample.shape)
