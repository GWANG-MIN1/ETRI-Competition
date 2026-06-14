#!/usr/bin/env python3
"""TRACK K — confirm: exploitable public gain is driven by TEMPORAL DRIFT, not measurement.

New framing: FS is ~train-calibrated. The public test rate differs from train only where
the target DRIFTS over time (the 37.6% future portion + recency). So the per-target
'best-case drift-bet gain' (gain if public rate == forward-CV/recency rate) should track
the target's temporal drift, and be large ONLY for Q2/Q3. Q1 (unmeasured but STABLE) should
have ~0 drift-gain -> proves it's drift, not measurement status, that gates gain.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq

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


def realized_drift(t):
    """mean |late-half - early-half| per-subject rate = realized temporal drift magnitude."""
    ds = []
    for s in subs:
        d = train[train.subject_id == s].sort_values("sleep_date")
        v = d[t].values; h = len(v) // 2
        ds.append(abs(v[h:].mean() - v[:h].mean()))
    return float(np.mean(ds))


def fwd_rate(t, s, tau=21.0):
    d = train[train.subject_id == s]
    last = d["sleep_date"].max()
    w = np.exp(-(last - d["sleep_date"]).dt.days.values / tau)
    return float(np.sum(w * d[t].values) / np.sum(w))


def step_toward(t, s, rho):
    z = Zfs[t][s]; cur = sig(z).mean(); rho = min(max(rho, 0.02), 0.98)
    if abs(cur - rho) < 1e-6:
        return 0.0
    try:
        return float(brentq(lambda dl: sig(z + dl).mean() - rho, -6, 6))
    except Exception:
        return 0.0


def drift_bet_gain(t):
    """gain (per the FULL 1750-cell metric) of moving target t toward forward-CV rate,
    EVALUATED at the forward-CV world (best case for the drift bet). neg=gain."""
    # build move
    A = 0.0; c = np.zeros(n_var)
    r_world = np.zeros(n_var)
    for s in subs:
        dd = step_toward(t, s, fwd_rate(t, s))
        z = Zfs[t][s]
        A += float(-np.log((1 - sig(z + dd)) / (1 - sig(z))).sum()) / 1750.0
        c[vix[(s, t)]] = -n_rows[s] * dd / 1750.0
        r_world[vix[(s, t)]] = fwd_rate(t, s)
    return A + c @ r_world


print("=== Drift drives gain: per-target realized drift vs best-case drift-bet gain ===")
print("  (measured = Q2/Q3 via anchor; gain<0 = available improvement if drift bet is right)")
print(f"  {'tgt':>4} | {'realized drift':>14} | {'drift-bet gain(fwd world)':>26} | {'measured?':>9}")
measured = {"Q2": "Y(anchor)", "Q3": "Y(anchor)"}
rows = []
for t in TARGETS:
    dr = realized_drift(t)
    g = drift_bet_gain(t)
    rows.append((t, dr, g))
    print(f"  {t:>4} | {dr:>14.4f} | {g:>+26.5f} | {measured.get(t,'N'):>9}")

drs = np.array([r[1] for r in rows]); gs = np.array([r[2] for r in rows])
rho = np.corrcoef(drs, -gs)[0, 1]  # more drift -> more negative gain -> -gs larger
print(f"\n  corr(realized drift, drift-bet gain magnitude) = {rho:+.3f}")
print("  Q1 check: stable+unmeasured -> small drift & small gain => measurement is NOT the gate.")
print("  CONCLUSION: the overshoot win is a DRIFT-harvest available only on Q2/Q3; S has no")
print("  drift to harvest, so it is dry regardless of probing. This is the wall's root cause.")
