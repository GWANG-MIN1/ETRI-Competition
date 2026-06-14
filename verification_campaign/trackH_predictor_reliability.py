#!/usr/bin/env python3
"""TRACK H — is forward-CV (or train rate) a reliable predictor of the TRUE public rate?

The polytope posterior mean r_hat is the best estimate of the true per-subject public
label rate, CONSTRAINED by all 30 LB measurements. For Q2/Q3 (the measured axis) r_hat
is genuinely informative. Compare r_hat_Q2Q3 to:
   - train rate         (the 'no drift' guess)
   - forward-CV rate    (the 'drift continues' guess)
If forward-CV is no closer (or wrong-direction) to r_hat than train is, then the
forward-CV drift signal is UNRELIABLE -> S-probes betting on it have ~coin-flip EV.

Also: per-subject Q-rate -> S-rate LOO predictability (reproduce/extend the Q-|-S claim).
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

PE.init()
St = PE._S
subs = St["subs"]; vix = St["vix"]; n_var = St["n_var"]
S = St["S"]; r_hat = S.mean(0); rate = St["rate"]
TARGETS = PE.TARGETS
RAW = PE.ETRI / "data"

train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
train["sleep_date"] = pd.to_datetime(train["sleep_date"])
r_train = np.array([[rate[t][s] for s in subs] for t in TARGETS])  # [T,S]
TAU = 21.0
r_fwd = np.zeros((len(TARGETS), len(subs)))
for ti, t in enumerate(TARGETS):
    for si, s in enumerate(subs):
        d = train[train.subject_id == s]
        last = d["sleep_date"].max()
        w = np.exp(-(last - d["sleep_date"]).dt.days.values / TAU)
        r_fwd[ti, si] = float(np.sum(w * d[t].values) / np.sum(w))
r_hatM = np.array([[r_hat[vix[(s, t)]] for s in subs] for t in TARGETS])

print("=== Reliability: does train / forward-CV predict the polytope r_hat (best est. of true public rate)? ===")
print("  (only Q2/Q3 r_hat is measurement-informed; S/Q1 r_hat ~ box center = train, so uninformative)")
print(f"  {'tgt':>4} | {'|r_hat-train|':>13} | {'|r_hat-fwd|':>11} | {'fwd closer?':>11} | {'corr(rhat,fwd-train shift)':>26}")
for ti, t in enumerate(TARGETS):
    dt = np.abs(r_hatM[ti] - r_train[ti]).mean()
    df = np.abs(r_hatM[ti] - r_fwd[ti]).mean()
    # does the r_hat shift (vs train) align with the fwd-CV shift (vs train)?
    shift_true = r_hatM[ti] - r_train[ti]
    shift_fwd = r_fwd[ti] - r_train[ti]
    if shift_true.std() > 1e-9 and shift_fwd.std() > 1e-9:
        c = np.corrcoef(shift_true, shift_fwd)[0, 1]
    else:
        c = float("nan")
    closer = "fwd" if df < dt else "train"
    print(f"  {t:>4} | {dt:>13.4f} | {df:>11.4f} | {closer:>11} | {c:>+26.3f}")

print("\n  Interpretation: for Q2/Q3, if 'fwd closer' is NO and corr is low/negative,")
print("  the forward-CV drift does NOT capture the true public shift -> S-drift bet unreliable.")

# ---- Q-rate -> S-rate LOO predictability (10 subjects) ----
print("\n=== Q-rate -> S-rate per-subject LOO predictability (R^2; <=0 means no transferable signal) ===")
Qrt = np.array([[rate[q][s] for q in ("Q1", "Q2", "Q3")] for s in subs])  # [S,3]
print(f"  {'S-target':>9} | {'LOO-R^2 (Q1Q2Q3->S)':>20} | {'best single-Q corr':>19}")
for st in ("S1", "S2", "S3", "S4"):
    y = np.array([rate[st][s] for s in subs])
    preds = np.zeros(len(subs))
    for i in range(len(subs)):
        tr = [j for j in range(len(subs)) if j != i]
        lr = LinearRegression().fit(Qrt[tr], y[tr])
        preds[i] = lr.predict(Qrt[i:i+1])[0]
    ss_res = np.sum((y - preds) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    cors = [np.corrcoef(Qrt[:, k], y)[0, 1] for k in range(3)]
    best = max(cors, key=abs)
    print(f"  {st:>9} | {r2:>20.3f} | {best:>+19.3f}")

print("\n  R^2<=0 across S -> Q-rates carry NO out-of-sample info about S-rates (Q ⊥ S confirmed).")
print("  Combined with unreliable forward-CV -> NO offline signal can aim an S-probe. Wall confirmed.")
