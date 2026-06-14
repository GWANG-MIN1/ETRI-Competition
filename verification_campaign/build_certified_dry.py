#!/usr/bin/env python3
"""DRY validation of the certified candidates (NO file saved — per user 'no CSV until asked').

Builds FS + certified delta for: overshoot x0.8 (safe-margin) and the constrained-optimal
(EV-best certified). Runs the upload-safety checklist (shape/keys/range/null/changed/date-std)
and reports per-subject Q2/Q3 moves + hash. Confirms submission-readiness without writing.
"""
from __future__ import annotations
from pathlib import Path
import sys, hashlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

KEYS = PE.KEYS; TARGETS = PE.TARGETS; EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    PE.init()
    vix = PE._S["vix"]; subs = PE._S["subs"]; d_over = PE._S["d_over"]
    fs = pd.read_csv(PE.ETRI / "submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv").sort_values(KEYS).reset_index(drop=True)
    sample = pd.read_csv(PE.ETRI / "data" / "ch2026_submission_sample.csv").sort_values(KEYS).reset_index(drop=True)
    fsub = fs["subject_id"].values
    base = fs[TARGETS].to_numpy(float)

    # Published candidate only. (The 'constrained-optimal' variant depended on a generated
    # outputs/*.npy artifact that is gitignored and not shipped; see save_overshoot_x08.py.)
    deltas = {
        "overshoot_x0.8 (safe-margin certified)": d_over * 0.8,
    }
    for name, d in deltas.items():
        out = fs[KEYS].copy()
        for t in TARGETS:
            z = logit(fs[t].values).copy()
            for s in subs:
                z[fsub == s] += d[vix[(s, t)]]
            out[t] = sigmoid(z)
        prob = out[TARGETS].to_numpy(float)
        digest = hashlib.sha1(np.round(prob, 12).tobytes()).hexdigest()[:8]
        dd = np.abs(prob - base)
        ok_shape = out.shape == (250, 10)
        ok_keys = out[KEYS].reset_index(drop=True).equals(sample[KEYS].reset_index(drop=True))
        ok_null = int(out[TARGETS].isnull().sum().sum()) == 0
        ok_range = bool(prob.min() > 0 and prob.max() < 1)
        print(f"\n=== {name}  (hash {digest}) ===")
        print(f"  [1] shape (250,10): {out.shape} -> {ok_shape}")
        print(f"  [2] keys==sample: {ok_keys}   [3] null==0: {ok_null}   [4] in (0,1): [{prob.min():.4f},{prob.max():.4f}] -> {ok_range}")
        print(f"  [5] vs FS: mean|d| {dd.mean():.5f} max|d| {dd.max():.4f} changed cells {int((dd>1e-12).sum())}")
        chg = [t for i, t in enumerate(TARGETS) if dd[:, i].max() > 1e-12]
        print(f"  [6] targets changed: {chg}  (Q2 mean {fs['Q2'].mean():.4f}->{out['Q2'].mean():.4f}, Q3 {fs['Q3'].mean():.4f}->{out['Q3'].mean():.4f})")
        print(f"  [7] UPLOAD-SAFE: {ok_shape and ok_keys and ok_null and ok_range}  (NOT saved — dry run)")
        print(f"  per-subject Q2 move: " + " ".join(f"{s}:{d[vix[(s,'Q2')]]:+.2f}" for s in subs))


if __name__ == "__main__":
    main()
