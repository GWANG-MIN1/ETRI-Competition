#!/usr/bin/env python3
"""TRACK M — is FS's own per-subject calibration improvable on the certified axis?

Distinct from toward-r_hat/forward-CV (trackA3b/C). Test FS recalibration variants as
per-subject level moves through worst-loose:
  - toward TRAIN rate           (pull each subject's FS level to its train base rate)
  - toward LAST-OBS rate        (most recent train value per subject; S is stable -> last obs may be best)
  - de-shrink (amplify FS's per-subject deviation from cohort mean by 1.15x)
  - shrink   (pull toward cohort mean 0.85x)
For each target-group. If none gives negative worst-loose, FS is already certified-optimal and
no recalibration beats the overshoot. (Handoff: FS de-shrunk = public-calibrated; shrink=mirage.)
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog, brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

PE.init()
St = PE._S
subs = St["subs"]; vix = St["vix"]; n_var = St["n_var"]; n_rows = St["n_rows"]
Zfs = St["Zfs"]; rate = St["rate"]
TARGETS = PE.TARGETS
RAW = PE.ETRI / "data"
train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
train["sleep_date"] = pd.to_datetime(train["sleep_date"])


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def last_obs_rate(t, s, k=5):
    d = train[train.subject_id == s].sort_values("sleep_date")
    return float(d[t].values[-k:].mean())


def step_toward(t, s, rho):
    z = Zfs[t][s]; cur = sig(z).mean(); rho = min(max(rho, 0.02), 0.98)
    if abs(cur - rho) < 1e-6:
        return 0.0
    try:
        return float(brentq(lambda dl: sig(z + dl).mean() - rho, -6, 6))
    except Exception:
        return 0.0


def gt(d):
    A = 0.0; c = np.zeros(n_var)
    for t in TARGETS:
        for s in subs:
            dd = d[vix[(s, t)]]
            if dd == 0:
                continue
            z = Zfs[t][s]
            A += float(-np.log((1 - sig(z + dd)) / (1 - sig(z))).sum()) / 1750.0
            c[vix[(s, t)]] = -n_rows[s] * dd / 1750.0
    return A, c


A_l, b_l, bnd_l = St["polytope"](0.30, 0.45, 3.0)


def worst(d):
    A, c = gt(d)
    r = linprog(-c, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
    return (A + (-r.fun)) if r.success else 1.0


def build(kind, targets):
    d = np.zeros(n_var)
    for t in targets:
        cohort = np.mean([rate[t][s] for s in subs])
        for s in subs:
            z = Zfs[t][s]; cur = sig(z).mean()
            if kind == "train":
                d[vix[(s, t)]] = step_toward(t, s, rate[t][s])
            elif kind == "lastobs":
                d[vix[(s, t)]] = step_toward(t, s, last_obs_rate(t, s))
            elif kind == "deshrink":
                d[vix[(s, t)]] = step_toward(t, s, cur + 0.15 * (cur - cohort))
            elif kind == "shrink":
                d[vix[(s, t)]] = step_toward(t, s, cur - 0.15 * (cur - cohort))
    return d


groups = {"Q2Q3": ["Q2", "Q3"], "S1-4": ["S1", "S2", "S3", "S4"], "ALL": list(TARGETS)}
print("FS recalibration variants -> worst-loose (negative = a certified improvement on FS).\n")
print(f"  {'variant':>12} | " + " ".join(f"{g:>10}" for g in groups))
for kind in ("train", "lastobs", "deshrink", "shrink"):
    cells = []
    for g, ts in groups.items():
        w = worst(build(kind, ts))
        cells.append(f"{w:>+10.5f}")
    print(f"  {kind:>12} | " + " ".join(cells))

print(f"\n  reference overshoot x0.8 worst-loose = {worst(St['d_over']*0.8):+.5f}")
print("  If all variants are >=0, FS calibration is already certified-optimal; only the")
print("  anchor-measured Q2/Q3 overshoot improves it. (No free recalibration lever.)")
