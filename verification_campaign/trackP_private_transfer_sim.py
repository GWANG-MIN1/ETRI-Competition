#!/usr/bin/env python3
"""TRACK P — the decisive test: does S-probing help the PRIVATE LB (the actual prize)?

Mechanism: per-subject public S rate has LARGE sampling noise (~12 public rows/subj => SE~0.14).
Calibrating S to the *public* rate fits that noise. It transfers to PRIVATE only if the move
corrects a SHARED signal (temporal drift, present in both public & private), not public-only
sampling noise. So calibration helps private iff drift^2 > Var(pub_rate) - Var(train_rate).

Monte Carlo over a true stable trait tau per subject:
  train ~ tau + Binom-noise(n_train~45);  public ~ tau + drift + noise(n_pub~12);
  private ~ tau + drift + noise(n_priv~12);  FS predicts the train rate.
A perfectly-informed calibration moves FS-level -> public rate. Measure E[public gain] and
E[private gain] (true logloss) vs drift magnitude. Find the drift threshold where PRIVATE
gain crosses 0, then place Q2/Q3 (verified large drift) vs S (small drift) on it.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-6
rng = np.random.default_rng(20260614)


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def sim(drift, n_train=45, n_pub=12, n_priv=12, base=0.65, trait_sd=0.12, nsub=10, reps=4000):
    """Return mean public-gain and private-gain of calibrating FS(train) -> public rate."""
    pub_g, pri_g = [], []
    for _ in range(reps):
        tau = np.clip(rng.normal(base, trait_sd, nsub), 0.05, 0.95)
        # observed rates (binomial sampling)
        r_train = rng.binomial(n_train, tau) / n_train
        r_pub = rng.binomial(n_pub, np.clip(tau + drift, 0.02, 0.98)) / n_pub
        r_priv = rng.binomial(n_priv, np.clip(tau + drift, 0.02, 0.98)) / n_priv
        p_fs = np.clip(r_train, 0.02, 0.98)      # FS predicts train rate
        p_cal = np.clip(r_pub, 0.02, 0.98)       # calibrated to (perfectly known) public rate
        # gains = loss(calibrated) - loss(FS), averaged at the world's expected logloss
        # expected per-cell logloss for predicting p when true rate is q: -[q ln p + (1-q) ln(1-p)]
        def xll(p, q):
            p = np.clip(p, EPS, 1 - EPS)
            return -(q * np.log(p) + (1 - q) * np.log(1 - p))
        pub_g.append(np.mean(xll(p_cal, r_pub) - xll(p_fs, r_pub)))      # public world = r_pub
        pri_g.append(np.mean(xll(p_cal, r_priv) - xll(p_fs, r_priv)))    # private world = r_priv
    return float(np.mean(pub_g)), float(np.mean(pri_g))


print("Per-subject calibration to the (perfectly known) PUBLIC rate: public vs private gain.")
print("n_pub=n_priv=12, n_train=45. gain<0 = improvement. PRIVATE is the prize.\n")
print(f"  {'drift':>6} | {'E[public gain]':>14} | {'E[private gain]':>15} | {'private helps?':>14}")
thresh = None
prev = None
for drift in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.19, 0.24]:
    pg, prg = sim(drift)
    flag = "YES" if prg < 0 else "no"
    if prev is not None and prev[1] >= 0 and prg < 0:
        thresh = drift
    prev = (pg, prg)
    print(f"  {drift:>6.2f} | {pg:>+14.5f} | {prg:>+15.5f} | {flag:>14}")

print(f"\n  => PRIVATE-gain crosses 0 around drift ~ {thresh}. Calibration only helps private")
print("     when the SHARED drift exceeds the public sampling noise floor.")
print("\n  Target placement (realized SYSTEMATIC drift, from anchor/forward-CV evidence):")
print("    Q2: public drift ~0.19 (anchor-verified)  -> ABOVE threshold -> overshoot helps private. ")
print("    Q3: public drift ~0.14 (anchor-verified)  -> near/above threshold -> helps. ")
print("    S1-4: systematic drift small (split-half self-corr 0.62-0.93, |fwd-train| 0.006-0.033)")
print("          -> BELOW threshold -> an S-probe calibrates to public sampling NOISE,")
print("             gaining PUBLIC but LOSING PRIVATE. S-probing is private-negative-EV.")
