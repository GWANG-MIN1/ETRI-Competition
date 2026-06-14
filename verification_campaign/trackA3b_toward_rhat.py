#!/usr/bin/env python3
"""TRACK A3b — 'move toward estimated rate' per cell-set, scale-swept.

Cleaner than A3: for each target group, build the per-(subject) logit step that
moves the FS subject-level toward the posterior-mean rate r_hat (the natural
'harvest' direction; for Q2/Q3 this ~ the overshoot). Sweep its scale and report
loose worst-case + post-mean + 18-setting favorability. If an S-group's toward-rate
move can reach robustly-negative worst-case, S is certifiable (NEW). If worst-case
stays >=0 at every scale, S is confirmed box-prior/dry.

Also compares against the over-anchor MIRAGE floor: a random orthogonal direction's
worst-case, to ensure any 'win' is not generic polytope credit.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linprog, brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

PE.init()
St = PE._S
subs = St["subs"]; vix = St["vix"]; n_var = St["n_var"]; n_rows = St["n_rows"]
Zfs = St["Zfs"]; S = St["S"]; r_hat = S.mean(0); d_over = St["d_over"]
TARGETS = PE.TARGETS


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def step_toward_rate(s, t, rho):
    """uniform logit shift delta so mean(sigmoid(z+delta)) = rho for subject s,target t."""
    z = Zfs[t][s]
    cur = sig(z).mean()
    rho = min(max(rho, 0.02), 0.98)
    if abs(cur - rho) < 1e-6:
        return 0.0
    f = lambda dl: sig(z + dl).mean() - rho
    try:
        return float(brentq(f, -6, 6))
    except Exception:
        return 0.0


# build the 'toward r_hat' unit direction per cell
d_dir = np.zeros(n_var)
for t in TARGETS:
    for s in subs:
        d_dir[vix[(s, t)]] = step_toward_rate(s, t, r_hat[vix[(s, t)]])


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


A_l, b_l, bnd_l = St["polytope"](0.30, 0.45, 3.0)


def worst_loose(d):
    A, c = gt(d)
    r = linprog(-c, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
    return (A + (-r.fun)) if r.success else 1.0


def worst18_fav(d):
    A, c = gt(d); ws = []
    for comp in (0.10, 0.20, 0.30):
        for bw in (0.25, 0.35, 0.45):
            for Z in (2.0, 3.0):
                Au, bu, bn = St["polytope"](comp, bw, Z)
                r = linprog(-c, A_ub=Au, b_ub=bu, bounds=bn, method="highs")
                if r.success:
                    ws.append(A + (-r.fun))
    ws = np.array(ws)
    return float(ws.max()), float((ws < 0).mean())


def postmean(d):
    A, c = gt(d)
    return float((S @ c + A).mean())


def restrict(cells):
    d = np.zeros(n_var)
    idx = list(cells)
    d[idx] = d_dir[idx]
    return d


groups = {
    "Q2Q3": {vix[(s, t)] for s in subs for t in ("Q2", "Q3")},
    "S1-4": {vix[(s, t)] for s in subs for t in ("S1", "S2", "S3", "S4")},
    "S1S2": {vix[(s, t)] for s in subs for t in ("S1", "S2")},
    "S3S4": {vix[(s, t)] for s in subs for t in ("S3", "S4")},
    "Q1": {vix[(s, t)] for s in subs for t in ("Q1",)},
    "ALL": set(range(n_var)),
}

SCALES = [0.25, 0.5, 0.75, 1.0, 1.25]
print("toward-r_hat move, scale-swept. worst-loose<0 & fav high = certifiable group.\n")
for name, cells in groups.items():
    base = restrict(cells)
    print(f"-- {name} (toward r_hat) --")
    best = None
    for sc in SCALES:
        d = base * sc
        wl = worst_loose(d); pm = postmean(d); w18, fav = worst18_fav(d)
        print(f"   scale {sc:.2f} | worst-loose {wl:+.5f} | worst18 {w18:+.5f} fav {fav:.2f} | post-mean {pm:+.5f}")
        if best is None or wl < best:
            best = wl
    print()

# mirage floor: random orthogonal directions' worst-loose (should be ~0 if certification is real)
print("=== MIRAGE control: random unit directions restricted to S1-4, worst-loose ===")
rng = np.random.default_rng(7)
Scells = list(groups["S1-4"])
wls = []
for k in range(15):
    d = np.zeros(n_var)
    u = rng.standard_normal(len(Scells)); u /= np.linalg.norm(u)
    d[Scells] = u * 0.5
    wls.append(worst_loose(d))
wls = np.array(wls)
print(f"  random S-dir worst-loose: mean {wls.mean():+.5f}  min {wls.min():+.5f}  max {wls.max():+.5f}")
print("  (if toward-r_hat S worst-loose is not far below this, it is mirage, not signal)")
