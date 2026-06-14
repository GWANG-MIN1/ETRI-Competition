#!/usr/bin/env python3
"""TRACK E — private robustness: how much does the overshoot-x0.8 conclusion depend on the
exact (noisy) 0.5619 anchor measurement?

Everything certified rests on ONE measurement (TRACK G). The public LB is a ~250-row logloss
= a noisy estimate of the true (and private) gain. Perturb the anchor value within plausible
sampling noise and recompute (a) the post-mean-optimal sigma*, (b) overshoot-x0.8 worst-loose.
If sigma* and the sign of the certified gain are stable across noise, the recommendation is
private-robust; if they swing, the 'certification' is really one noisy data point.

For per-subject-uniform moves gain is assignment-noise-free => public gain ~ private gain
(the 99.3% realized transfer). So the dominant private risk IS this measurement noise, not
public->private transfer. This brackets it.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

PE.init()
St = PE._S
subs = St["subs"]; vix = St["vix"]; n_var = St["n_var"]; n_rows = St["n_rows"]
Zfs = St["Zfs"]; d_over = St["d_over"]; rate = St["rate"]; named = St["named"]
TARGETS = PE.TARGETS
ANCHOR_I = len(named) - 1  # the 0.5619 anchor constraint (appended last)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


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


def loose_polytope_perturbed(danchor, comp=0.30, boxw=0.45, Zz=3.0):
    bnd = [(max(0.02, rate[t][s] - boxw), min(0.98, rate[t][s] + boxw)) for t in TARGETS for s in subs]
    AA, bb = [], []
    for i, (v, c0, d, sg) in enumerate(named):
        d_use = danchor if i == ANCHOR_I else d
        tol = 1e-5 + comp * abs(d_use) + Zz * sg
        AA.append(v); bb.append(d_use + tol - c0)
        AA.append(-v); bb.append(-(d_use - tol - c0))
    return np.array(AA), np.array(bb), bnd


d_anchor0 = named[ANCHOR_I][2]  # = ANCHOR_LB - H057_LB ~ -0.00583
print(f"baseline anchor delta (vs H057) = {d_anchor0:+.5f}  (LB 0.56191 - 0.56775)")
print("\n=== sigma* and overshoot-x0.8 worst-loose under perturbed anchor measurement ===")
print("  (perturb the 0.5619 public score by +-noise; 250-row logloss SE ~ 0.0003-0.0006)")
print(f"  {'anchor LB':>10} | {'sigma* (post-min)':>17} | {'x0.8 worst-loose':>16} | {'sign certified?':>15}")
H057_LB = PE.H057_LB
for noise in [-0.0006, -0.0003, 0.0, +0.0003, +0.0006]:
    d_anch = d_anchor0 + noise
    A_l, b_l, bnd_l = loose_polytope_perturbed(d_anch)

    # use the FIXED posterior for post-mean (mild approximation), but recompute worst on perturbed loose
    def worst(d):
        A, c = gt(d)
        r = linprog(-c, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
        return (A + (-r.fun)) if r.success else 1.0
    # post-mean uses a quick posterior proxy: gain at the constraint-consistent chebyshev center
    # (cheap): solve max-margin point, use as r-hat-ish. Good enough for sigma* location.
    lob = np.array([b[0] for b in bnd_l]); hib = np.array([b[1] for b in bnd_l])
    Aall = np.vstack([A_l, np.eye(n_var), -np.eye(n_var)]); ball = np.concatenate([b_l, hib, -lob])
    nm = np.linalg.norm(Aall, axis=1, keepdims=True)
    ch = linprog(np.r_[np.zeros(n_var), -1.0], A_ub=np.hstack([Aall, nm]), b_ub=ball,
                 bounds=[(None, None)] * n_var + [(0, None)], method="highs")
    rc = ch.x[:n_var]
    best_sg = None
    for sg in np.arange(0.4, 1.21, 0.05):
        A, c = gt(d_over * sg)
        pm = A + c @ rc
        if best_sg is None or pm < best_sg[1]:
            best_sg = (sg, pm)
    w08 = worst(d_over * 0.8)
    print(f"  {H057_LB + d_anch:>10.5f} | {best_sg[0]:>17.2f} | {w08:>+16.5f} | "
          f"{'YES' if w08 < 0 else 'NO':>15}")

print("\n  Stable sigma* (~0.7-0.85) and worst-loose<0 across the noise band => private-robust.")
print("  Swinging sign => the certified improvement is within measurement noise (treat as ~anchor).")
