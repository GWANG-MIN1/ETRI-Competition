"""Verify the red-team's load-bearing empirical claims against disk."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs/raw"
SUBS = ROOT / "outputs/submissions"
TARGETS = ["Q1","Q2","Q3","S1","S2","S3","S4"]
S_OBJ = ["S1","S2","S3","S4"]; Q_SUB = ["Q1","Q2","Q3"]

lbl = pd.read_csv(RAW/"ch2026_metrics_train.csv")
ps = lbl.groupby("subject_id")[TARGETS].mean()
ps["S_obj"] = ps[S_OBJ].mean(axis=1); ps["Q_sub"] = ps[Q_SUB].mean(axis=1)

print("=== per-subject means + mismatch (Q1 - S_obj) ===")
ps["mismatch_Q1_Sobj"] = ps["Q1"] - ps["S_obj"]
print(ps[["Q1","S_obj","Q_sub","mismatch_Q1_Sobj"]].round(3).to_string())

print("\n=== CLAIM 1: corr(S_obj, Q_sub) across 10 subjects + bootstrap CI ===")
r = np.corrcoef(ps["S_obj"], ps["Q_sub"])[0,1]
r1 = np.corrcoef(ps["S1"], ps["Q1"])[0,1]
print(f"  corr(S_obj,Q_sub) = {r:.3f} | corr(S1,Q1) = {r1:.3f}")
rng = np.random.default_rng(0); boot = []
idx = np.arange(len(ps))
for _ in range(5000):
    b = rng.choice(idx, size=len(idx), replace=True)
    so, qs = ps["S_obj"].values[b], ps["Q_sub"].values[b]
    if np.std(so) > 1e-9 and np.std(qs) > 1e-9:
        boot.append(np.corrcoef(so, qs)[0,1])
boot = np.array(boot)
print(f"  subject-block bootstrap 95% CI = [{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}]  (crosses 0: {np.percentile(boot,2.5)<0<np.percentile(boot,97.5)})")

print("\n=== CLAIM (Q tight / S wide): per-target between-subject SD ===")
for t in TARGETS:
    print(f"  {t}: SD={ps[t].std():.3f}")

print("\n=== CLAIM 2: was S234 rerank mean-preserving? (per-target mean vs best) ===")
best = pd.read_csv(SUBS/"submission_FINAL_Q2recency_plus_Q3recency.csv")
rr = SUBS/"submission_ROBUST_S234_rerank_w025.csv"
if rr.exists():
    r2 = pd.read_csv(rr)
    for t in TARGETS:
        d = r2[t].mean() - best[t].mean()
        flag = "  <-- NOT mean-preserving" if abs(d) > 5e-4 else ""
        print(f"  {t}: best {best[t].mean():.4f} -> rerank {r2[t].mean():.4f}  (delta {d:+.4f}){flag}")
else:
    print("  rerank file missing")

print("\n=== within-subject Q1~S1 point-biserial (per subject, n nights) ===")
for s, g in lbl.groupby("subject_id"):
    if g["S1"].nunique() > 1 and g["Q1"].nunique() > 1:
        rr_ = np.corrcoef(g["Q1"], g["S1"])[0,1]
        print(f"  {s}: n={len(g)} corr(Q1,S1)={rr_:+.3f}")
