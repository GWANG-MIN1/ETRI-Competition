#!/usr/bin/env python3
"""TRACK A2 — measurement-coverage per target (the real 'which axes are pinned' test).

The polytope's LINEAR constraints come from the `named` list: each is v·r in
[d-tol, d+tol], where v[(s,t)] is how much ledger-move m perturbs cell (s,t).
A target t is PINNED only if some constraints have large |v| on its cells.
If all constraints have ~0 energy on S cells -> S-rate is box-only -> uncertifiable.

We compute, per target:
  - total constraint energy   sum_m ||v_m restricted to t||^2
  - max single-constraint |v| on any t-cell
  - effective # of measurements that move t (|v_m,t| above noise)
Also: per-target the *tightest* one-sided rate bound the polytope implies vs the
box, by LP-maximising / minimising each cell's r and seeing if constraints bite.
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
S = PE._S
subs = S["subs"]; vix = S["vix"]; named = S["named"]; n_var = S["n_var"]
TARGETS = PE.TARGETS
rate = S["rate"]

# ---- 1. constraint energy per target ----
print("=== Constraint energy per target (how much the 30 ledger moves perturb each axis) ===")
print(f"{'tgt':>4} | {'tot energy':>11} | {'max|v|cell':>11} | {'#moves>1e-4':>11} | {'mean rate':>9}")
tgt_cells = {t: [vix[(s, t)] for s in subs] for t in TARGETS}
for t in TARGETS:
    cells = tgt_cells[t]
    energy = 0.0
    maxv = 0.0
    nmoves = 0
    for v, c0, d, sg in named:
        vt = v[cells]
        e = float(np.sum(vt ** 2))
        energy += e
        maxv = max(maxv, float(np.max(np.abs(vt))))
        if np.max(np.abs(vt)) > 1e-4:
            nmoves += 1
    mr = np.mean([rate[t][s] for s in subs])
    print(f"{t:>4} | {energy:>11.3e} | {maxv:>11.3e} | {nmoves:>11} | {mr:>9.3f}")

# ---- 2. polytope-implied per-cell rate interval vs box (does any constraint bite?) ----
# Use the loose polytope. For each cell, LP max and min r_cell.
A_l, b_l, bnd_l = S["polytope"](0.30, 0.45, 3.0)
lob = np.array([b[0] for b in bnd_l]); hib = np.array([b[1] for b in bnd_l])

print("\n=== Per-target: how much the constraints TIGHTEN the box (avg interval shrink) ===")
print("  (shrink>0 means linear constraints bind beyond the box -> that axis is measured)")
print(f"{'tgt':>4} | {'box width':>10} | {'poly width':>10} | {'avg shrink':>10} | {'max shrink':>10}")
for t in TARGETS:
    box_w = []
    poly_w = []
    for s in subs:
        j = vix[(s, t)]
        c = np.zeros(n_var); c[j] = 1.0
        rmax = linprog(-c, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
        rmin = linprog(c, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
        if rmax.success and rmin.success:
            pw = (-rmax.fun) - (rmin.fun)
            box_w.append(hib[j] - lob[j])
            poly_w.append(pw)
    box_w = np.array(box_w); poly_w = np.array(poly_w)
    shrink = box_w - poly_w
    print(f"{t:>4} | {box_w.mean():>10.4f} | {poly_w.mean():>10.4f} | {shrink.mean():>10.4f} | {shrink.max():>10.4f}")

print("\nInterpretation: targets with ~0 constraint energy AND ~0 shrink are box-prior only")
print("=> the polytope cannot certify ANY directional gain on them (handoff S/Q1 claim).")
