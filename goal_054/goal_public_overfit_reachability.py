#!/usr/bin/env python3
"""GOAL — is public-LB 0.54 a real score or a public-overfit artifact?

The public LB is ~125 rows (~12/subject). With small per-subject samples, even simple
per-subject-rate calibration FIT TO THE PUBLIC SUBSET drives public logloss far below the
true floor (small-n rates are extreme -> low entropy), and full per-row probing drives it ->0.
So 'public 0.54' may not reflect transferable skill. Quantify the public-overfit floor:
for each target, subsample ~12 rows/subject (public-like), fit per-subject rate ON that subset,
score ON THE SAME subset. Mean over targets = the per-subject-calibration public-overfit floor.
If that is already <= ~0.54, then top-public scores are reachable by overfitting the small public
set and say little about the PRIVATE (paper) ranking, where we established ~0.5619 is the floor.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\박광민\Documents\Codex\etri_team\data")
tr = pd.read_csv(RAW / "ch2026_metrics_train.csv")
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6
rng = np.random.default_rng(0)


def Hbin(r):
    r = np.clip(r, EPS, 1 - EPS)
    return -(r * np.log(r) + (1 - r) * np.log(1 - r))


# public-like subsample: ~12 rows/subject; fit per-subject rate on it; logloss on same (overfit)
def overfit_floor(t, n_pub=12, smooth=0.0, reps=300):
    lls = []
    for _ in range(reps):
        ll = []
        for s, g in tr.groupby("subject_id"):
            y = g[t].values
            if len(y) > n_pub:
                y = rng.choice(y, n_pub, replace=False)
            r = (y.sum() + smooth) / (len(y) + 2 * smooth)   # optionally smoothed
            ll.append(np.mean(-(y * np.log(np.clip(r, EPS, 1 - EPS)) + (1 - y) * np.log(np.clip(1 - r, EPS, 1 - EPS)))))
        lls.append(np.mean(ll))
    return float(np.mean(lls))


print("=== public-overfit floor: per-subject-rate calibration FIT to a ~12-row public subset ===")
print("  (this is the WEAKEST overfit; full per-row probing pushes public -> 0)")
print(f"  {'tgt':>4} | {'overfit floor (raw)':>19} | {'overfit floor (smoothed)':>24}")
raws, sms = [], []
for t in TARGETS:
    a = overfit_floor(t, smooth=0.0)
    b = overfit_floor(t, smooth=1.0)   # add-1 smoothing = more realistic submitted prob
    raws.append(a); sms.append(b)
    print(f"  {t:>4} | {a:>19.4f} | {b:>24.4f}")
print(f"  {'MEAN':>4} | {np.mean(raws):>19.4f} | {np.mean(sms):>24.4f}")

print("\n=== context ===")
print(f"  per-subject-rate public-overfit floor (raw):      {np.mean(raws):.4f}")
print(f"  per-subject-rate public-overfit floor (smoothed): {np.mean(sms):.4f}")
print(f"  our current public:                               0.5615")
print(f"  top-5 public (reported):                          ~0.5400")
print(f"  our established PRIVATE floor (certified):        ~0.5619")
print()
print("  If the overfit floor <= ~0.54, then 0.54-on-PUBLIC is reachable by fitting the small public")
print("  subset (per-subject, or per-row with probing) and does NOT imply 0.54 on PRIVATE. The paper")
print("  prize is PRIVATE; chasing public 0.54 by overfitting is expected to HURT the private rank.")
print("  If the overfit floor is >> 0.54, then top teams have genuine signal we must find.")
