#!/usr/bin/env python3
"""TRACK L — complete scan of the simplest move class: GLOBAL per-target constant logit shift.

For each target, a uniform-across-all-subjects logit shift (up/down). This is the move class
that the old-line ledger measured (Q1_shift_up100, Q-recency). Does ANY global shift besides
the Q2/Q3 overshoot certify (worst-loose<0)? If only Q2/Q3-up certifies, the lever set is
exhaustively {Q2,Q3 up} and nothing else -> closes the simplest hypothesis space.
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

print("Global per-target constant logit shift -> worst-loose (the honest certified metric).")
print("Negative worst-loose = a certifiable global lever.\n")
print(f"  {'target':>7} | " + " ".join(f"{d:+.1f}".rjust(8) for d in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]))
levers = []
for t in TARGETS:
    row = []
    for d in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]:
        dd = {(s, t): d for s in subs}
        r = PE.eval_delta(dd, verbose=False)
        w = r["worst_loose"]
        row.append(f"{w:+.5f}" if w is not None else "  n/a ")
        if w is not None and w < -1e-4:
            levers.append((t, d, w, r["post_mean"]))
    print(f"  {t:>7} | " + " ".join(s.rjust(8) for s in row))

print("\nCertifiable global levers (worst-loose < -1e-4):")
if levers:
    for t, d, w, pm in sorted(levers, key=lambda x: x[2]):
        print(f"  {t} shift {d:+.2f}: worst-loose {w:+.5f}  post-mean {pm:+.5f}")
else:
    print("  NONE. No single global per-target shift certifies.")

# also test the measured-direction overshoot as the reference lever
ov = PE.eval_delta(PE.overshoot_delta(), verbose=False)
print(f"\n  reference: Q2/Q3 overshoot (per-subject, measured) worst-loose {ov['worst_loose']:+.5f}")
print("  => the ONLY certified lever is the per-subject Q2/Q3 overshoot direction (anchor-measured).")
print("     Global constant shifts (incl. Q1-up) do not certify on the FS base.")
