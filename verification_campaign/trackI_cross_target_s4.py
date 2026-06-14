#!/usr/bin/env python3
"""TRACK I — investigate the ONE positive cross-target lead: Q3-rate -> S4-rate.

Q3<->S4 per-subject corr = -0.846, LOO-R^2 +0.27 (only positive S). The anchor pins
the true public Q3 rate (r_hat_Q3). IF the Q3->S4 subject relationship is stable (not a
1-subject artifact), we could infer the public S4 rate from the anchor and aim an S4 move
WITHOUT spending an S-probe slot. Test rigorously:
  1. robustness of Q3->S4 corr to dropping each subject (is it one outlier?)
  2. is it a subject-stable TRAIT? split each subject's train rows in half, corr(Q3_h1, S4_h2)
  3. what public S4 rate does f(r_hat_Q3) imply, and what is that S4 move's gamble profile?
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog, brentq
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

PE.init()
St = PE._S
subs = St["subs"]; vix = St["vix"]; n_var = St["n_var"]; n_rows = St["n_rows"]
Zfs = St["Zfs"]; S = St["S"]; r_hat = S.mean(0); rate = St["rate"]
TARGETS = PE.TARGETS
RAW = PE.ETRI / "data"
train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
train["sleep_date"] = pd.to_datetime(train["sleep_date"])

q3 = np.array([rate["Q3"][s] for s in subs])
s4 = np.array([rate["S4"][s] for s in subs])

print("=== 1. Q3->S4 corr robustness (drop each subject) ===")
base = np.corrcoef(q3, s4)[0, 1]
print(f"  full corr = {base:+.3f}")
drops = []
for i, s in enumerate(subs):
    idx = [j for j in range(len(subs)) if j != i]
    c = np.corrcoef(q3[idx], s4[idx])[0, 1]
    drops.append((s, c))
for s, c in sorted(drops, key=lambda x: x[1]):
    print(f"    drop {s}: corr {c:+.3f}")
print(f"  range after any single drop: [{min(c for _,c in drops):+.3f}, {max(c for _,c in drops):+.3f}]")

print("\n=== 2. Is Q3->S4 a subject-STABLE trait? (within-subject split-half cross corr) ===")
# per subject: early-half Q3 rate vs late-half S4 rate, and vice versa
q3a, q3b, s4a, s4b = [], [], [], []
for s in subs:
    d = train[train.subject_id == s].sort_values("sleep_date")
    h = len(d) // 2
    q3a.append(d["Q3"].values[:h].mean()); q3b.append(d["Q3"].values[h:].mean())
    s4a.append(d["S4"].values[:h].mean()); s4b.append(d["S4"].values[h:].mean())
q3a, q3b, s4a, s4b = map(np.array, (q3a, q3b, s4a, s4b))
print(f"  corr(Q3_earlyhalf, S4_latehalf)  = {np.corrcoef(q3a, s4b)[0,1]:+.3f}")
print(f"  corr(Q3_latehalf,  S4_earlyhalf) = {np.corrcoef(q3b, s4a)[0,1]:+.3f}")
print(f"  corr(Q3 split-half self)         = {np.corrcoef(q3a, q3b)[0,1]:+.3f}  (trait stability of Q3)")
print(f"  corr(S4 split-half self)         = {np.corrcoef(s4a, s4b)[0,1]:+.3f}  (trait stability of S4)")
print("  cross-half Q3->S4 staying strong & negative => stable trait link, not noise.")

print("\n=== 3. Anchor-implied public S4 rate, and the implied S4 move's gamble ===")
# fit S4 ~ Q3 on train rates (LOO already +0.27). Predict public S4 from r_hat_Q3.
r_hat_q3 = np.array([r_hat[vix[(s, "Q3")]] for s in subs])
lr = LinearRegression().fit(q3.reshape(-1, 1), s4)
s4_pred_public = lr.predict(r_hat_q3.reshape(-1, 1))
print(f"  fit: S4_rate = {lr.coef_[0]:+.3f}*Q3_rate + {lr.intercept_:+.3f}")
print(f"  {'subj':>5} | {'train S4':>8} | {'r_hat Q3':>8} | {'pred pub S4':>11} | {'implied S4 shift':>16}")
for i, s in enumerate(subs):
    print(f"  {s:>5} | {s4[i]:>8.3f} | {r_hat_q3[i]:>8.3f} | {s4_pred_public[i]:>11.3f} | {s4_pred_public[i]-s4[i]:>+16.3f}")


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def step_toward(s, rho):
    z = Zfs["S4"][s]; cur = sig(z).mean(); rho = min(max(rho, 0.02), 0.98)
    if abs(cur - rho) < 1e-6:
        return 0.0
    try:
        return float(brentq(lambda dl: sig(z + dl).mean() - rho, -6, 6))
    except Exception:
        return 0.0


d = np.zeros(n_var)
for i, s in enumerate(subs):
    d[vix[(s, "S4")]] = step_toward(s, s4_pred_public[i])


def gt(dd):
    A = 0.0; c = np.zeros(n_var)
    for t in TARGETS:
        for s in subs:
            v = dd[vix[(s, t)]]
            if v == 0:
                continue
            z = Zfs[t][s]
            A += float(-np.log((1 - sig(z + v)) / (1 - sig(z))).sum()) / 1750.0
            c[vix[(s, t)]] = -n_rows[s] * v / 1750.0
    return A, c


A_l, b_l, bnd_l = St["polytope"](0.30, 0.45, 3.0)
A, c = gt(d)
g_train = A + c @ np.array([rate[t][s] for t in TARGETS for s in subs])
g_post = float((S @ c + A).mean())
rw = linprog(-c, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
worst = A + (-rw.fun) if rw.success else None
rb = linprog(c, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
best = A - (-rb.fun) if rb.success else None
print(f"\n  S4->anchor-implied move gamble: train {g_train:+.5f} | best {best:+.5f} | "
      f"worst {worst:+.5f} | post {g_post:+.5f}")
print("  Verdict needs: stable trait (step2) + bounded worst. If worst >> |best|, still a bad bet.")
