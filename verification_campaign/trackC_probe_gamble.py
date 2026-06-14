#!/usr/bin/env python3
"""TRACK C — honest gamble characterization of candidate probes on the UNMEASURED axes.

A per-subject-uniform move's gain is an exact linear functional of the per-subject
TRUE rate r (assignment-noise-free) -> identical for public and private subsets.
So we can read its gain under any hypothesised rate world:
  - r = TRAIN rate           (the 'no drift' world; FS ~ calibrated here -> gain ~0)
  - r = FORWARD-CV rate      (recency-weighted; the 'drift continues' bet)
  - worst-loose              (adversarial corner over the loose polytope)
  - post-mean over posterior (MIRAGE-flagged: credits generic moves; not decisive)

This quantifies each S/Q1 probe's bet: 'if drift continues -> gain X; if adversarial
-> lose Y; if no drift -> ~Z'. The measured Q2/Q3 overshoot x0.8 is the reference.
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
Zfs = St["Zfs"]; S = St["S"]; d_over = St["d_over"]; rate = St["rate"]
TARGETS = PE.TARGETS
RAW = PE.ETRI / "data"


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---- rate worlds ----
train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
train["sleep_date"] = pd.to_datetime(train["sleep_date"])
r_train = np.zeros(n_var)
r_fwd = np.zeros(n_var)
TAU = 21.0
for t in TARGETS:
    for s in subs:
        d = train[train.subject_id == s]
        r_train[vix[(s, t)]] = d[t].mean()
        last = d["sleep_date"].max()
        w = np.exp(-(last - d["sleep_date"]).dt.days.values / TAU)
        r_fwd[vix[(s, t)]] = float(np.sum(w * d[t].values) / np.sum(w))


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


def characterize(d, label):
    A, c = gt(d)
    g_train = A + c @ r_train
    g_fwd = A + c @ r_fwd
    g_post = float((S @ c + A).mean())
    r = linprog(-c, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
    worst = (A + (-r.fun)) if r.success else None
    rb = linprog(c, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
    best = (A - (-rb.fun)) if rb.success else None  # A + c@r_best
    print(f"  {label:>26} | train {g_train:+.5f} | fwd-CV {g_fwd:+.5f} | "
          f"best {best:+.5f} | worst {worst:+.5f} | post {g_post:+.5f}")
    return dict(train=g_train, fwd=g_fwd, post=g_post, worst=worst, best=best)


def step_toward(s, t, rho, frac=1.0):
    z = Zfs[t][s]; cur = sig(z).mean(); rho = min(max(cur + frac * (rho - cur), 0.02), 0.98)
    if abs(cur - rho) < 1e-6:
        return 0.0
    try:
        return float(brentq(lambda dl: sig(z + dl).mean() - rho, -6, 6))
    except Exception:
        return 0.0


def toward_world(targets, world, frac=1.0):
    d = np.zeros(n_var)
    for t in targets:
        for s in subs:
            d[vix[(s, t)]] = step_toward(s, t, world[vix[(s, t)]], frac)
    return d


print("Gain of each move under hypothesised rate worlds (neg=improve).")
print("'fwd-CV' = the drift-continues bet; 'worst' = adversarial; 'post' = MIRAGE-flagged.\n")
print("--- REFERENCE (measured Q2/Q3 axis) ---")
characterize(d_over * 0.8, "overshoot x0.8 (measured)")

print("\n--- S-block probes (toward forward-CV rate = drift bet) ---")
characterize(toward_world(["S2"], r_fwd), "S2 -> fwd-CV")
characterize(toward_world(["S1", "S2"], r_fwd), "S1S2 -> fwd-CV")
characterize(toward_world(["S1", "S2", "S3", "S4"], r_fwd), "S1-4 -> fwd-CV")
characterize(toward_world(["S1", "S2", "S3", "S4"], r_fwd, frac=0.5), "S1-4 -> halfway fwd-CV")

print("\n--- Q1 probe (unmeasured subjective) ---")
characterize(toward_world(["Q1"], r_fwd), "Q1 -> fwd-CV")

print("\n--- direction sanity: do S targets even DRIFT in train? (fwd-CV vs train rate) ---")
print(f"  {'tgt':>4} | {'mean|fwd-train|':>15} | {'subjects drifting >0.05':>24}")
for t in TARGETS:
    diffs = np.array([r_fwd[vix[(s, t)]] - r_train[vix[(s, t)]] for s in subs])
    ndr = int(np.sum(np.abs(diffs) > 0.05))
    print(f"  {t:>4} | {np.mean(np.abs(diffs)):>15.4f} | {ndr:>24}")

print("\nKey: a probe is a POSITIVE bet only if fwd-CV gain is clearly negative AND the")
print("worst-case downside is bounded/small. post-mean negativity alone = mirage (ignore).")
