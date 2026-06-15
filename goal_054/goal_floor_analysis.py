#!/usr/bin/env python3
"""GOAL grounding — information-theoretic logloss floors: is 0.54 reachable?

Decompose the achievable mean-logloss into:
  F_global  = predict each target's GLOBAL train rate          (no subject, no per-row info)
  F_subject = predict each subject's train rate (CV, honest)   (perfect per-subject calibration,
              NO per-row info) -> the ceiling that pure level/calibration work can reach.
  OOF_unified = a real per-row model's leakage-free OOF        (uses per-row info)
  current best public = 0.5615333 (overshoot x0.8 on the jackpot/FS base)
If 0.5615 < F_subject, the base already exploits per-row discrimination, and 0.54 requires
EVEN MORE transferable per-row signal. Quantify exactly how much (the per-row headroom needed).
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

CORE = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(CORE))
import hsjepa_core as H  # noqa: E402

EPS = 1e-6
TARGETS = H.TARGETS
m = H.load()  # KEYS + y_* + p_* (unified OOF)


def bll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def Hbin(r):
    r = np.clip(r, EPS, 1 - EPS)
    return float(-(r * np.log(r) + (1 - r) * np.log(1 - r)))


print("=== Per-target floors (mean logloss; lower=better) ===")
print(f"  {'tgt':>4} | {'globalrate':>10} | {'subj-cal(CV)':>12} | {'unified OOF':>11} | {'subj rate spread':>16}")
rows = []
for t in TARGETS:
    y = m[f"y_{t}"].values
    # global rate floor (in-sample entropy of the global rate)
    g = y.mean(); fg = Hbin(g)
    # per-subject calibration floor, honest 5-fold interleaved CV (predict held-out by train-fold subj rate)
    sublls = []
    for tr, va in H.interleaved_folds(m, 5, seed=11):
        smean = m.iloc[tr].groupby("subject_id")[f"y_{t}"].mean().to_dict()
        gm = m.iloc[tr][f"y_{t}"].mean()
        subj = m["subject_id"].values[va]
        pred = np.array([smean.get(s, gm) for s in subj])
        sublls.append(bll(y[va], pred))
    fs_ = float(np.mean(sublls))
    oof = bll(y, m[f"p_{t}"].values)
    spread = float(np.std(list(m.groupby("subject_id")[f"y_{t}"].mean())))
    rows.append((t, fg, fs_, oof))
    print(f"  {t:>4} | {fg:>10.4f} | {fs_:>12.4f} | {oof:>11.4f} | {spread:>16.3f}")

mg = np.mean([r[1] for r in rows]); ms = np.mean([r[2] for r in rows]); mo = np.mean([r[3] for r in rows])
print(f"  {'MEAN':>4} | {mg:>10.4f} | {ms:>12.4f} | {mo:>11.4f}")

print("\n=== Where does the current best sit? ===")
CUR = 0.5615333471
print(f"  global-rate floor (no info)            : {mg:.4f}")
print(f"  per-subject calibration floor (no perrow): {ms:.4f}   <- best achievable w/ ZERO per-row signal")
print(f"  unified per-row OOF (real model)       : {mo:.4f}")
print(f"  CURRENT BEST (public)                  : {CUR:.4f}")
print(f"  TARGET                                 : 0.5400")
print()
print(f"  current is BELOW per-subject floor by  : {ms - CUR:+.4f}  "
      f"({'per-row signal IS exploited' if CUR < ms else 'not even at calib floor'})")
print(f"  gap current -> 0.54                    : {CUR - 0.54:+.4f}  (must be found in per-row signal)")
print(f"  0.54 below per-subject calib floor by  : {ms - 0.54:+.4f}  "
      f"=> 0.54 REQUIRES strong transferable per-row discrimination (level/calibration alone caps at {ms:.3f})")

# crude per-row oracle estimate: if rows were perfectly separable within the model's AUC ceiling,
# logloss floor ~ depends on irreducible label noise. Estimate via best single model OOF as a proxy.
print("\n=== Interpretation ===")
print("  - logloss below the per-subject floor can ONLY come from per-row discrimination that")
print("    correctly orders within-subject rows. The jackpot did this once (0.5932->0.5677).")
print("  - 0.54 needs the per-row component to improve by ~", round(CUR - 0.54, 4), "mean logloss")
print("    beyond the current jackpot base, AND it must TRANSFER (public->private).")
