#!/usr/bin/env python3
"""TRACK A — per-cell certification scan.

For every (subject,target) cell, sweep a single-cell logit shift and find the
certified worst-case gain (min over the loose polytope). This maps EXACTLY which
cells the existing 30-measurement polytope can certify a gain on.

Claim under test (handoff §3): only Q2/Q3 are polytope-pinned (certifiable);
S/Q1 are box-prior (uncertifiable). If true, only Q2/Q3 cells should yield a
negative certified worst-case; S/Q1 cells should bottom out near 0.

Output: per-cell best certified worst-case + the delta that achieves it.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

PE.init()
subs = PE.subjects
TARGETS = PE.TARGETS

# sweep grid of single-cell logit shifts (both directions)
GRID = [-0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]

print("Per-cell CERTIFIED worst-case gain (most-negative achievable over grid).")
print("Negative = polytope certifies a real improvement on that cell.\n")

# accumulate per target
by_target = {t: [] for t in TARGETS}
best_overall = []
for t in TARGETS:
    for s in subs:
        best_w = 0.0
        best_d = 0.0
        best_pm = 0.0
        for d in GRID:
            r = PE.eval_delta({(s, t): d}, label="", verbose=False)
            if r["worst_loose"] is not None and r["worst_loose"] < best_w:
                best_w = r["worst_loose"]
                best_d = d
                best_pm = r["post_mean"]
        by_target[t].append((s, best_w, best_d, best_pm))
        if best_w < -1e-5:
            best_overall.append((s, t, best_w, best_d, best_pm))

print(f"{'target':>7} | {'#cells cert<0':>13} | {'best cert worst':>16} | {'sum cert worst':>15}")
for t in TARGETS:
    cells = by_target[t]
    ncert = sum(1 for (_, w, _, _) in cells if w < -1e-5)
    bestw = min(w for (_, w, _, _) in cells)
    sumw = sum(w for (_, w, _, _) in cells)
    print(f"{t:>7} | {ncert:>13} | {bestw:>+16.5f} | {sumw:>+15.5f}")

print("\nAll cells with certifiable gain (worst-case < -1e-5), sorted:")
best_overall.sort(key=lambda x: x[2])
for s, t, w, d, pm in best_overall:
    print(f"  {s}.{t:>2}  delta {d:+.2f}  cert-worst {w:+.5f}  post-mean {pm:+.5f}")

print(f"\nTOTAL certifiable cells: {len(best_overall)} / {len(subs)*len(TARGETS)}")
print("Per-target breakdown of certifiable cells:")
for t in TARGETS:
    n = sum(1 for x in best_overall if x[1] == t)
    print(f"  {t}: {n}")
