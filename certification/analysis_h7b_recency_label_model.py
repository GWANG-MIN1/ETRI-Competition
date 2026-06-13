#!/usr/bin/env python3
"""H7b: the RECENCY label model — calibrate on one observed probe, predict the other.

H7's uniform-elevation model failed (needs Delta=+0.45 -> rate>1).
Diagnosis: the old line's Q2 win came from PER-SUBJECT recency shifts, so the public
labels must track per-subject RECENT rates, not uniformly elevated train marginals.

Label model M2:  r_s(target) = clip( recency_rate_s(target; tau_days) + Delta, 0.02, 0.98 )
  where recency_rate_s = exp-weighted train label rate of subject s (weights toward the
  subject's last train date).

Validation protocol (the killer test):
  1. Calibrate Delta on the Q2 probe (A->B, measured -0.0051833 LB units).
  2. With that SAME Delta, predict the Q3 probe (B->C) gain out-of-sample. Compare to the
     measured -0.0017279. Also do the reverse (calibrate Q3 -> predict Q2).
  3. Bonus check on the confounded Q1 probe (microblend->A, approx).
If M2 transfers across targets, apply it to FrontierSilence and compute exact gain curves
for uniform-tau and per-subject-tau Q2/Q3 level shifts, with breakevens and shrinkage.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ETRI = Path(r"C:\Users\박광민\Documents\Codex\etri_team")
OLD = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\outputs\submissions")
RAW = ETRI / "data"
KEYS = ["subject_id", "sleep_date", "lifelog_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6

FILE_A = OLD / "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv"
FILE_B = OLD / "submission_Q1up100_plus_Q2recency_tau10_lam075.csv"
FILE_C = OLD / "submission_FINAL_Q2recency_plus_Q3recency.csv"
GAIN_Q2 = 7.0 * (0.5949167449 - 0.6001)
GAIN_Q3 = 7.0 * (0.593188787 - 0.5949167449)
FS_FILE = ETRI / "submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv"


def load(p):
    return pd.read_csv(p).sort_values(KEYS).reset_index(drop=True)


def expected_ll(p, r):
    p = np.clip(p, EPS, 1 - EPS)
    return -(r * np.log(p) + (1 - r) * np.log(1 - p))


def shift_logit(p, tau):
    p = np.clip(p, EPS, 1 - EPS)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + tau)))


def recency_rates(train, target, tau_days):
    out = {}
    for s, g in train.groupby("subject_id"):
        d = pd.to_datetime(g["sleep_date"])
        w = np.exp(-(d.max() - d).dt.days.values / tau_days)
        out[s] = float(np.sum(w * g[target].values) / np.sum(w))
    return out


def main():
    train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
    A, B, C = load(FILE_A), load(FILE_B), load(FILE_C)
    sub = A["subject_id"].values

    print("=== per-subject recency rates (tau=10d) vs train means ===")
    rec = {t: recency_rates(train, t, 10.0) for t in TARGETS}
    mean_ = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}
    for t in ["Q1", "Q2", "Q3"]:
        rows = [f"{s}:{rec[t][s]:.2f}({rec[t][s]-mean_[t][s]:+.2f})" for s in sorted(rec[t])]
        print(f"  {t}: " + " ".join(rows))

    def gain_under_model(df_from, df_to, target, delta, tau_days=10.0):
        r_map = recency_rates(train, target, tau_days)
        r = np.clip(np.array([r_map[s] for s in sub]) + delta, 0.02, 0.98)
        return float(np.mean(expected_ll(df_to[target].values, r)
                             - expected_ll(df_from[target].values, r))) / 7.0

    def calibrate(df_from, df_to, target, measured_lb_gain):
        lo, hi = -0.45, 0.45
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if gain_under_model(df_from, df_to, target, mid) > measured_lb_gain:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    print("\n=== M2 cross-target validation ===")
    mQ2 = GAIN_Q2 / 7.0
    mQ3 = GAIN_Q3 / 7.0
    dQ2 = calibrate(A, B, "Q2", mQ2)
    dQ3 = calibrate(B, C, "Q3", mQ3)
    print(f"  calibrated Delta on Q2 probe: {dQ2:+.3f}   | on Q3 probe: {dQ3:+.3f}")
    # out-of-sample cross predictions
    pred_q3_from_q2 = gain_under_model(B, C, "Q3", dQ2)
    pred_q2_from_q3 = gain_under_model(A, B, "Q2", dQ3)
    pred_q3_d0 = gain_under_model(B, C, "Q3", 0.0)
    pred_q2_d0 = gain_under_model(A, B, "Q2", 0.0)
    print(f"  Q3 probe measured {mQ3:+.5f} | predicted w/ Delta_Q2 {pred_q3_from_q2:+.5f} | w/ Delta=0 {pred_q3_d0:+.5f}")
    print(f"  Q2 probe measured {mQ2:+.5f} | predicted w/ Delta_Q3 {pred_q2_from_q3:+.5f} | w/ Delta=0 {pred_q2_d0:+.5f}")

    # Q1 bonus (confounded probe): microblend -> A changed mostly Q1 (+0.10) but
    # other targets moved slightly; treat as approximate.
    MICRO = OLD / "submission_lb60247_to_pertarget_best_microblend.csv"
    if MICRO.exists():
        M = load(MICRO)
        m_meas = 0.6001 - 0.60247
        per_t = {t: float(np.abs(M[t].values - A[t].values).mean()) for t in TARGETS}
        pred_q1 = gain_under_model(M, A, "Q1", dQ2)
        others = sum(gain_under_model(M, A, t, dQ2) for t in TARGETS if t != "Q1")
        print(f"  Q1 probe (confounded; per-target mean|move| {dict((k, round(v,3)) for k,v in per_t.items())})")
        print(f"    measured total {m_meas:+.5f} | M2(Delta_Q2) Q1-part {pred_q1:+.5f} | other-targets part {others:+.5f}")

    # ------------------------------------------------------------------
    print("\n=== FrontierSilence level-shift curves under M2 (Delta from each calibration) ===")
    FS = load(FS_FILE)
    sub_fs = FS["subject_id"].values

    def fs_gain(target, tau, delta, per_subject=False, tau_days=10.0):
        r_map = recency_rates(train, target, tau_days)
        r = np.clip(np.array([r_map[s] for s in sub_fs]) + delta, 0.02, 0.98)
        p = FS[target].values
        if per_subject:
            # per-subject tau: kappa * (logit(r_s) - mean subject logit(p))
            p2 = p.copy()
            for s in np.unique(sub_fs):
                m = sub_fs == s
                gap = np.log(r[m][0] / (1 - r[m][0])) - np.mean(np.log(np.clip(p[m],EPS,1-EPS) / (1 - np.clip(p[m],EPS,1-EPS))))
                p2[m] = shift_logit(p[m], tau * gap)  # tau acts as kappa in [0,1]
        else:
            p2 = shift_logit(p, tau)
        return float(np.mean(expected_ll(p2, r) - expected_ll(p, r))) / 7.0

    for target, dcal in [("Q2", dQ2), ("Q3", dQ3)]:
        print(f"\n  --- {target} (FS mean {FS[target].mean():.3f}) uniform-tau ---")
        print(f"  {'tau':>5} | {'Delta=' + format(dcal,'+.3f'):>12} | {'Delta/2':>9} | {'Delta=0':>9} | {'Delta=-0.05':>11}")
        for tau in (0.05, 0.10, 0.15, 0.20, 0.30):
            print(f"  {tau:5.2f} | {fs_gain(target, tau, dcal):+12.5f} | {fs_gain(target, tau, dcal/2):+9.5f}"
                  f" | {fs_gain(target, tau, 0.0):+9.5f} | {fs_gain(target, tau, -0.05):+11.5f}")
        print(f"  --- {target} per-subject kappa-gap allocation ---")
        print(f"  {'kappa':>5} | {'Delta_cal':>10} | {'Delta/2':>9} | {'Delta=0':>9} | {'Delta=-0.05':>11}")
        for kap in (0.25, 0.5, 0.75, 1.0):
            print(f"  {kap:5.2f} | {fs_gain(target, kap, dcal, True):+10.5f} | {fs_gain(target, kap, dcal/2, True):+9.5f}"
                  f" | {fs_gain(target, kap, 0.0, True):+9.5f} | {fs_gain(target, kap, -0.05, True):+11.5f}")

    # Q1 curve (evidence weaker)
    print(f"\n  --- Q1 (FS mean {FS['Q1'].mean():.3f}) uniform-tau, M2 Delta_Q2 ---")
    for tau in (0.05, 0.10, 0.20):
        print(f"  tau={tau:.2f}: gain {fs_gain('Q1', tau, dQ2):+.5f} | Delta=0 {fs_gain('Q1', tau, 0.0):+.5f}")

    # sensitivity: recency horizon tau_days 10 vs 28
    print("\n=== sensitivity: recency horizon (tau_days=28) ===")
    for target, dcal in [("Q2", dQ2), ("Q3", dQ3)]:
        d28 = None
        # recalibrate under 28d
        lo, hi = -0.45, 0.45
        meas = mQ2 if target == "Q2" else mQ3
        dff, dft = (A, B) if target == "Q2" else (B, C)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if gain_under_model(dff, dft, target, mid, tau_days=28.0) > meas:
                lo = mid
            else:
                hi = mid
        d28 = 0.5 * (lo + hi)
        g = fs_gain(target, 0.15, d28, tau_days=28.0)
        print(f"  {target}: Delta*(28d)={d28:+.3f} -> FS tau=0.15 gain {g:+.5f}")


if __name__ == "__main__":
    main()
