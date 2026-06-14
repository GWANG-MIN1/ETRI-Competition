#!/usr/bin/env python3
"""TRACK D — certified ensemble blend (CEILING briefing seed #1, 'likely unexplored').

Blend the decorrelated good submissions (logit-mean). Decompose the blend's move vs FS
into per-subject LEVEL (transferable, certifiable) vs within-subject RESIDUAL (the
variance-reduction part, a per-row lottery that does NOT transfer). Then:
  - eval the LEVEL component through the polytope (worst-loose). If ~0/positive, the
    blend offers no certifiable gain -> its apparent benefit is non-transferable residual.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

PE.init()
St = PE._S
subs = St["subs"]; vix = St["vix"]
TARGETS = PE.TARGETS
ETRI = PE.ETRI
KEYS = PE.KEYS
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


SEARCH = [ETRI, ETRI / "analysis_outputs"]


def load(name):
    for base in SEARCH:
        p = base / name
        if p.exists():
            return pd.read_csv(p).sort_values(KEYS).reset_index(drop=True)
    raise FileNotFoundError(name)


FS = pd.read_csv(ETRI / PE.FS_FILE).sort_values(KEYS).reset_index(drop=True)
# decorrelated good models (different families, all LB 0.5677-0.5763)
pool_names = [
    "submission_h057_q2row_fullvector_state_7cde1a77_uploadsafe.csv",
    "submission_e247_featnn1_nn_smooth_sum_top34_f1ff7e86.csv",
    "submission_mixmin_0c916bb4.csv",
    "submission_e95_hardtail_541e3973.csv",
]
pool = []
for nm in pool_names:
    try:
        pool.append(load(nm))
        print(f"  loaded {nm.split('_')[1]}")
    except Exception as e:
        print(f"  (skip {nm}: {e})")
fsub = FS["subject_id"].values

# blends: FS + equal-weight logit mean with the pool, varying pool weight
print("=== Blend level-vs-residual decomposition + certified worst-case ===")
print(f"  {'blend':>28} | {'||level||':>9} | {'||resid||':>9} | {'lvl/total':>9} | {'worst-loose':>11} | {'post':>9}")
for w in [0.2, 0.35, 0.5]:
    # blended logit per cell
    blend = FS[KEYS].copy()
    levmove = {}
    tot_lvl = 0.0; tot_res = 0.0
    delta_lvl = {}
    for t in TARGETS:
        zfs = logit(FS[t].values)
        zpool = np.mean([logit(d[t].values) for d in pool], axis=0)
        zbl = (1 - w) * zfs + w * zpool
        move = zbl - zfs  # per-row logit move
        # per-subject level = mean move within subject; residual = move - level
        for s in subs:
            msk = fsub == s
            lev = move[msk].mean()
            delta_lvl[(s, t)] = float(lev)
            tot_lvl += (lev ** 2) * msk.sum()
            tot_res += float(np.sum((move[msk] - lev) ** 2))
    lvln = np.sqrt(tot_lvl); resn = np.sqrt(tot_res)
    res = PE.eval_delta(delta_lvl, label="", verbose=False)
    frac = lvln / (lvln + resn)
    print(f"  FS+{w:.2f}*pool(level only) ".rjust(28)[:28] +
          f" | {lvln:>9.4f} | {resn:>9.4f} | {frac:>9.3f} | {res['worst_loose']:>+11.5f} | {res['post_mean']:>+9.5f}")

print("\n  If level worst-loose >= 0 (uncertified) while residual norm dominates, the blend's")
print("  benefit is the non-transferable per-row variance-reduction lottery -> wall confirmed.")
print("  (Per handoff S/N law: dense level transfers, sparse residual reshape does not.)")
