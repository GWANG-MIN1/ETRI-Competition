#!/usr/bin/env python3
"""TRACK Q — rigorous per-target SYSTEMATIC temporal drift (replaces noise-inflated split-half).

The private-negative S-probe verdict hinges on S having small systematic drift (<0.12 threshold).
Split-half |late-early| is noise-inflated. Decompose within-subject temporal structure properly:
  1. Linear-trend slope per subject (label ~ normalized time); permutation test vs shuffled-time.
  2. Time-block variance decomposition: between-block (systematic) vs within-block (iid noise).
  3. Implied train->test-future drift = slope * (time gap to test future), the threshold-relevant #.
Compare Q2/Q3 (should show real drift) vs S1-4 (should be ~noise). Bootstrap CIs over subjects.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\박광민\Documents\Codex\etri_team\data")
train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
train["sleep_date"] = pd.to_datetime(train["sleep_date"])
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
subs = sorted(train.subject_id.unique())
rng = np.random.default_rng(0)


def subject_series(t):
    out = {}
    for s in subs:
        d = train[train.subject_id == s].sort_values("sleep_date")
        y = d[t].values.astype(float)
        # normalized time 0..1
        days = (d["sleep_date"] - d["sleep_date"].min()).dt.days.values.astype(float)
        x = days / max(days.max(), 1)
        out[s] = (x, y)
    return out


def slope(x, y):
    if x.std() < 1e-9:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


print("=== 1. Per-subject linear-trend slope (label over normalized time 0->1) ===")
print("  slope ~ total drift across a subject's observed span. permutation p = P(|shuffled| >= |obs|)")
print(f"  {'tgt':>4} | {'mean|slope|':>11} | {'median slope':>12} | {'perm p(mean|slope|)':>20} | {'frac subj |slope|>0.2':>21}")
drift_est = {}
for t in TARGETS:
    ser = subject_series(t)
    slopes = np.array([slope(x, y) for s, (x, y) in ser.items()])
    obs = np.mean(np.abs(slopes))
    # permutation: shuffle time within subject
    perm = []
    for _ in range(500):
        sh = []
        for s, (x, y) in ser.items():
            yp = rng.permutation(y)
            sh.append(slope(x, yp))
        perm.append(np.mean(np.abs(sh)))
    p = float(np.mean(np.array(perm) >= obs))
    frac = float(np.mean(np.abs(slopes) > 0.2))
    drift_est[t] = obs
    print(f"  {t:>4} | {obs:>11.3f} | {np.median(slopes):>+12.3f} | {p:>20.3f} | {frac:>21.2f}")

print("\n=== 2. Time-block variance decomposition (3 blocks): systematic vs iid-noise ===")
print("  ratio = between-block var / expected-iid var. ~1 => pure noise (no drift); >>1 => systematic.")
print(f"  {'tgt':>4} | {'between-block var':>16} | {'expected iid var':>16} | {'ratio (drift signal)':>20}")
for t in TARGETS:
    bvars, evars = [], []
    for s in subs:
        d = train[train.subject_id == s].sort_values("sleep_date")
        y = d[t].values.astype(float)
        n = len(y)
        if n < 9:
            continue
        blocks = np.array_split(y, 3)
        bm = np.array([b.mean() for b in blocks])
        bvar = bm.var()
        p = y.mean()
        # expected variance of a block mean under iid = p(1-p)/blocksize; var of 3 block-means ~ that/...
        bs = np.mean([len(b) for b in blocks])
        evar = p * (1 - p) / bs * (2 / 3)  # var of block means around grand mean under iid
        bvars.append(bvar); evars.append(evar)
    ratio = np.sum(bvars) / (np.sum(evars) + 1e-12)
    print(f"  {t:>4} | {np.mean(bvars):>16.4f} | {np.mean(evars):>16.4f} | {ratio:>20.2f}")

print("\n=== 3. Threshold comparison (private-transfer threshold ~ 0.12 from trackP) ===")
print("  systematic drift estimate (mean|slope|, ~ full-span drift) vs 0.12 threshold:")
for t in TARGETS:
    verdict = "ABOVE -> private-exploitable" if drift_est[t] > 0.12 else "below -> private-noise"
    star = " <<<" if t in ("Q2", "Q3") else ""
    print(f"  {t}: {drift_est[t]:.3f}   {verdict}{star}")
print("\n  Q2/Q3 expected ABOVE (drift real); S1-4 expected below (confirms private-negative S-probe).")
