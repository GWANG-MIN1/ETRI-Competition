#!/usr/bin/env python3
"""TRACK Q2 — per-target private-EV of an optimal probe-informed calibration, using REAL drift.

Refine the S-probe verdict: estimate each target's SYSTEMATIC drift SD (block decomposition,
noise-subtracted) + whether the drift is a GLOBAL shared direction (capturable by recency, no
probe) or PER-SUBJECT idiosyncratic (needs a probe). Then run the trackP private-transfer Monte
Carlo with each target's real (drift_sd, n_pub) to get E[private gain] of probe-informed calibration.
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
rng = np.random.default_rng(7)
EPS = 1e-6

# rows-per-subject in TEST (from FS) -> public ~half
TEST_N = {"id01": 27, "id02": 32, "id03": 21, "id04": 27, "id05": 21,
          "id06": 24, "id07": 30, "id08": 19, "id09": 27, "id10": 22}


def drift_sd_and_direction(t):
    """systematic drift SD (noise-subtracted, via block var) + global-direction consistency."""
    sysv, slopes = [], []
    for s in subs:
        d = train[train.subject_id == s].sort_values("sleep_date")
        y = d[t].values.astype(float)
        if len(y) < 9:
            continue
        blocks = np.array_split(y, 3)
        bm = np.array([b.mean() for b in blocks])
        bs = np.mean([len(b) for b in blocks])
        p = y.mean()
        evar = p * (1 - p) / bs * (2 / 3)
        sysv.append(max(0.0, bm.var() - evar))
        days = (d["sleep_date"] - d["sleep_date"].min()).dt.days.values.astype(float)
        x = days / max(days.max(), 1)
        slopes.append(float(np.polyfit(x, y, 1)[0]) if x.std() > 1e-9 else 0.0)
    drift_sd = float(np.sqrt(np.mean(sysv)))
    slopes = np.array(slopes)
    # direction consistency: |mean slope| / mean|slope| (1=global, 0=idiosyncratic)
    consist = abs(slopes.mean()) / (np.mean(np.abs(slopes)) + 1e-9)
    return drift_sd, consist, float(slopes.mean())


def xll(p, q):
    p = np.clip(p, EPS, 1 - EPS)
    return -(q * np.log(p) + (1 - q) * np.log(1 - p))


def private_ev(drift_sd, base=0.65, n_train=45, reps=6000):
    """E[private gain] of calibrating FS(train) -> perfectly-known public rate, where each
    subject's true test shift ~ N(0, drift_sd) (idiosyncratic drift) and public/private are
    independent ~12-row samples around tau+shift."""
    pub_g, pri_g = [], []
    for _ in range(reps):
        tau = np.clip(rng.normal(base, 0.12, len(subs)), 0.05, 0.95)
        shift = rng.normal(0, drift_sd, len(subs))            # per-subject systematic test drift
        npub = np.array([max(4, TEST_N[s] // 2) for s in subs])
        npriv = np.array([max(4, TEST_N[s] - TEST_N[s] // 2) for s in subs])
        r_train = rng.binomial(n_train, tau) / n_train
        q = np.clip(tau + shift, 0.02, 0.98)
        r_pub = rng.binomial(npub, q) / npub
        r_priv = rng.binomial(npriv, q) / npriv
        p_fs = np.clip(r_train, 0.02, 0.98)
        p_cal = np.clip(r_pub, 0.02, 0.98)
        pub_g.append(np.mean(xll(p_cal, r_pub) - xll(p_fs, r_pub)))
        pri_g.append(np.mean(xll(p_cal, r_priv) - xll(p_fs, r_priv)))
    return float(np.mean(pub_g)), float(np.mean(pri_g))


print("Per-target systematic drift + probe-informed calibration private-EV.")
print("(drift global => recency captures it free; idiosyncratic => needs a per-subject probe)\n")
print(f"  {'tgt':>4} | {'drift_sd':>8} | {'dir-consist':>11} | {'mean slope':>10} | "
      f"{'E[pub gain]':>11} | {'E[priv gain]':>12} | verdict")
for t in TARGETS:
    dsd, consist, ms = drift_sd_and_direction(t)
    pg, prg = private_ev(dsd)
    if prg < -0.0005:
        v = "PROBE helps private"
    elif prg < 0.0005:
        v = "~neutral private"
    else:
        v = "PROBE hurts private"
    star = " <<<" if t in ("Q2", "Q3") else ""
    print(f"  {t:>4} | {dsd:>8.3f} | {consist:>11.2f} | {ms:>+10.3f} | {pg:>+11.5f} | {prg:>+12.5f} | {v}{star}")

print("\n  dir-consist ~1 => one global direction (recency/forward-CV captures it without a probe).")
print("  Honest refinement: S2/S4 have REAL but small idiosyncratic drift; the private-EV verdict")
print("  per target tells whether a probe is worth it (negative E[priv gain] = worth probing).")
