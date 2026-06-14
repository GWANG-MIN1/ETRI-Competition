#!/usr/bin/env python3
"""TRACK F+G — refine the overshoot scaling & test its dependence on the anchor measurement.

F. Per-target sigma: is uniform overshoot*0.8 dominated by (sigma_Q2, sigma_Q3) separate,
   or by a per-subject scaling? Use the authoritative polytope_eval worst_loose + post_mean.
   (Scaling is collinear with the measured axis, so NOT subject to the §4B mirage.)

G. Leave-one-measurement-out: rebuild the polytope dropping each ledger row (esp. the 0.5619
   anchor) and recompute the overshoot's worst_loose. If dropping the anchor collapses the
   certification, the whole edifice rests on one measurement (fragile). If robust to LOO, solid.
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
Zfs = St["Zfs"]; S = St["S"]; d_over = St["d_over"]; named = St["named"]
TARGETS = PE.TARGETS


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


def worst_on(d, A_l, b_l, bnd_l):
    A, c = gt(d)
    r = linprog(-c, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
    return (A + (-r.fun)) if r.success else 1.0


def postmean(d):
    A, c = gt(d)
    return float((S @ c + A).mean())


# split overshoot into Q2 and Q3 components
d_q2 = np.zeros(n_var); d_q3 = np.zeros(n_var)
for s in subs:
    d_q2[vix[(s, "Q2")]] = d_over[vix[(s, "Q2")]]
    d_q3[vix[(s, "Q3")]] = d_over[vix[(s, "Q3")]]

A_l, b_l, bnd_l = St["polytope"](0.30, 0.45, 3.0)

print("=== F1. Uniform sigma fine sweep (authoritative posterior) ===")
print(f"  {'sigma':>6} | {'post-mean':>10} | {'worst-loose':>11}")
best_u = None
for sg in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1]:
    d = d_over * sg
    pm = postmean(d); wl = worst_on(d, A_l, b_l, bnd_l)
    print(f"  {sg:6.2f} | {pm:+10.5f} | {wl:+11.5f}")
    if best_u is None or pm < best_u[1]:
        best_u = (sg, pm, wl)
print(f"  -> uniform post-mean-opt sigma* = {best_u[0]} (pm {best_u[1]:+.5f}, worst {best_u[2]:+.5f})")

print("\n=== F2. Separate (sigma_Q2, sigma_Q3) grid (post-mean ; worst-loose) ===")
grid = [0.4, 0.6, 0.8, 1.0, 1.2]
best_s = None
print("        " + " ".join(f"Q3={g:.1f}" for g in grid))
for a in grid:
    row = []
    for b in grid:
        d = d_q2 * a + d_q3 * b
        pm = postmean(d); wl = worst_on(d, A_l, b_l, bnd_l)
        row.append(f"{pm:+.4f}")
        if (wl < 0) and (best_s is None or pm < best_s[2]):
            best_s = (a, b, pm, wl)
    print(f"  Q2={a:.1f} " + " ".join(row))
if best_s:
    print(f"  -> best CERTIFIED (worst<0) separate: Q2={best_s[0]} Q3={best_s[1]} pm {best_s[2]:+.5f} worst {best_s[3]:+.5f}")

print("\n=== F3. Does separate beat uniform 0.8? (both must keep worst<0) ===")
d_u08 = d_over * 0.8
print(f"  uniform x0.8: pm {postmean(d_u08):+.5f}  worst {worst_on(d_u08,A_l,b_l,bnd_l):+.5f}")

# ---- G. leave-one-measurement-out ----
print("\n=== G. Leave-one-measurement-out: overshoot x0.8 worst-loose dropping each ledger row ===")
# named is the list of constraints; rebuild loose polytope omitting index i
rate = St["rate"]


def polytope_drop(drop_idx, comp=0.30, boxw=0.45, Zz=3.0):
    bnd = [(max(0.02, rate[t][s] - boxw), min(0.98, rate[t][s] + boxw)) for t in TARGETS for s in subs]
    AA, bb = [], []
    for i, (v, c0, d, sg) in enumerate(named):
        if i == drop_idx:
            continue
        tol = 1e-5 + comp * abs(d) + Zz * sg
        AA.append(v); bb.append(d + tol - c0)
        AA.append(-v); bb.append(-(d - tol - c0))
    return np.array(AA), np.array(bb), bnd


full_w = worst_on(d_u08, A_l, b_l, bnd_l)
print(f"  full polytope worst-loose(x0.8) = {full_w:+.5f}")
print(f"  {'drop#':>6} | {'worst-loose':>11} | {'delta vs full':>13} | note")
# identify the anchor index (last named appended = the 0.5619 anchor)
anchor_idx = len(named) - 1
results = []
for i in range(len(named)):
    Ad, bd, bnd = polytope_drop(i)
    w = worst_on(d_u08, Ad, bd, bnd)
    results.append((i, w))
results.sort(key=lambda x: -abs(x[1] - full_w))
for i, w in results[:8]:
    note = "<-- 0.5619 ANCHOR" if i == anchor_idx else ""
    print(f"  {i:>6} | {w:+11.5f} | {w-full_w:+13.5f} | {note}")
print(f"  (showing 8 most-impactful drops of {len(named)} measurements)")
print(f"  worst-loose still <0 after dropping anchor? "
      f"{'YES' if dict(results)[anchor_idx] < 0 else 'NO'} "
      f"(value {dict(results)[anchor_idx]:+.5f})")
