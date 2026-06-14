#!/usr/bin/env python3
"""TRACK J — per-subject rate self-stability: how much TRANSFERABLE level even exists?

A per-subject level move only transfers if the per-subject rate is a STABLE TRAIT
(early-half rate predicts late-half rate). If a target's per-subject rate has near-zero
split-half self-correlation, there is barely any stable level to transfer -> the wall is
fundamentally a per-subject-signal-starvation, not just measurement coverage.

Also bootstraps the self-stability to gauge noise (10 subjects), and ranks targets by
(stability x measurement-gap) = OED value: a stable-but-unmeasured target is the best probe.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

PE.init()
St = PE._S
subs = St["subs"]; vix = St["vix"]
S = St["S"]; r_hat = S.mean(0); rate = St["rate"]
TARGETS = PE.TARGETS
RAW = PE.ETRI / "data"
train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
train["sleep_date"] = pd.to_datetime(train["sleep_date"])


def split_half_corr(t, seed=None):
    a, b = [], []
    rng = np.random.default_rng(seed) if seed is not None else None
    for s in subs:
        d = train[train.subject_id == s].sort_values("sleep_date")
        v = d[t].values
        if rng is not None:
            v = rng.permutation(v)  # random split (vs temporal)
        h = len(v) // 2
        a.append(v[:h].mean()); b.append(v[h:].mean())
    a, b = np.array(a), np.array(b)
    return np.corrcoef(a, b)[0, 1]


print("=== Per-subject rate self-stability (split-half corr; high=stable trait, transfers) ===")
print(f"  {'tgt':>4} | {'temporal split':>14} | {'random split':>12} | {'rand boot 90% CI':>20} | {'subj rate spread':>16}")
for t in TARGETS:
    ts = split_half_corr(t)
    boots = np.array([split_half_corr(t, seed=k) for k in range(200)])
    rs = float(np.mean(boots))
    lo, hi = np.percentile(boots, [5, 95])
    spread = float(np.std([rate[t][s] for s in subs]))
    print(f"  {t:>4} | {ts:>+14.3f} | {rs:>+12.3f} | [{lo:+.2f},{hi:+.2f}]{'':>8} | {spread:>16.3f}")

print("\n  random-split corr ~ true trait stability (temporal adds drift). High = real level.")

# measurement gap: how unconstrained is each target's rate (box width vs poly width from A2)
# reuse: targets with constraint but low stability = nothing to gain; stable + unmeasured = probe value
print("\n=== OED ranking: stable AND unmeasured = best probe target ===")
print("  (Q2/Q3 are measured (anchor); S/Q1 unmeasured. Stability from random split above.)")
measured = {"Q2": True, "Q3": True, "Q1": False, "S1": False, "S2": False, "S3": False, "S4": False}
rows = []
for t in TARGETS:
    rs = float(np.mean([split_half_corr(t, seed=k) for k in range(200)]))
    spread = float(np.std([rate[t][s] for s in subs]))
    # OED proxy: stability * spread * (1 if unmeasured else 0)
    val = max(rs, 0) * spread * (0 if measured[t] else 1)
    rows.append((t, rs, spread, measured[t], val))
rows.sort(key=lambda x: -x[4])
print(f"  {'tgt':>4} | {'stability':>9} | {'spread':>7} | {'measured':>8} | {'OED value (stab*spread)':>22}")
for t, rs, sp, me, val in rows:
    print(f"  {t:>4} | {rs:>+9.3f} | {sp:>7.3f} | {str(me):>8} | {val:>22.4f}")
print("\n  Highest OED value = the single most informative S/Q1 probe to spend a slot on.")
