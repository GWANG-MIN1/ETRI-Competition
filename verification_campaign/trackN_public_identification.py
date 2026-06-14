#!/usr/bin/env python3
"""TRACK N — quantify public-subset identifiability from the 30 LB measurements.

The public LB subset is a FIXED ~50% set of rows, SHARED across all 7 targets. Each LB
measurement is one (noisy) linear functional of the per-(subject,target) PUBLIC label rate
vector r_pub (70-dim). Questions:
  1. Effective rank: how many dims of r_pub do the 30 measurements actually pin (SVD vs noise)?
  2. Which targets live in the identified subspace? (does it touch S at all?)
  3. Decompose the Q2/Q3 public-vs-train gap into DRIFT (forward-CV) + COMPOSITION residual.
     If a shared composition factor exists, Q-measurements could inform the S public rate
     (same rows!) -> a NEW lever toward S without an S-probe.
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
subs = St["subs"]; vix = St["vix"]; n_var = St["n_var"]; named = St["named"]
S = St["S"]; r_hat = S.mean(0); rate = St["rate"]
TARGETS = PE.TARGETS
RAW = PE.ETRI / "data"

# ---- measurement matrix M (n_meas x 70), tolerances, deltas ----
M = np.array([v for (v, c0, d, sg) in named])        # rows = measurements
tol = np.array([1e-5 + 0.10 * abs(d) + 2.0 * sg for (v, c0, d, sg) in named])
dvec = np.array([d - c0 for (v, c0, d, sg) in named])  # measured value net of const
nmeas = M.shape[0]
print(f"measurement matrix: {M.shape}  (measurements x cells)")

# row-normalize by tolerance (so 'signal' = how tightly each constraint pins its direction)
Mn = M / tol[:, None]
U, sv, Vt = np.linalg.svd(Mn, full_matrices=False)
print("\n=== 1. Identifiability spectrum (singular values of tolerance-normalized M) ===")
print("  sv >> 1 => that direction is pinned tighter than measurement noise.")
print("  " + "  ".join(f"{s:.1f}" for s in sv))
neff = int(np.sum(sv > 1.0))
print(f"  effective identified dims (sv>1): {neff} / {n_var}")

print("\n=== 2. Target composition of the top identified directions (|Vt| mass per target) ===")
print(f"  {'dir(sv)':>10} | " + " ".join(f"{t:>5}" for t in TARGETS))
for k in range(min(8, len(sv))):
    vk = Vt[k]
    mass = {t: np.sqrt(sum(vk[vix[(s, t)]] ** 2 for s in subs)) for t in TARGETS}
    tot = sum(mass.values()) + 1e-12
    print(f"  {k}({sv[k]:>5.1f}) | " + " ".join(f"{mass[t]/tot:>5.2f}" for t in TARGETS))

# cumulative S-identifiability: project a unit S-only direction onto identified subspace
print("\n=== 3. How identified is each target? (fraction of a unit target-direction captured by sv>1 subspace) ===")
Vid = Vt[:neff]  # identified row-space
print(f"  {'target':>7} | {'identified frac':>15} | {'max single-cell id':>18}")
for t in TARGETS:
    fracs = []
    for s in subs:
        e = np.zeros(n_var); e[vix[(s, t)]] = 1.0
        proj = Vid @ e
        fracs.append(float(np.sum(proj ** 2)))  # ||P_id e||^2 in [0,1]
    print(f"  {t:>7} | {np.mean(fracs):>15.3f} | {np.max(fracs):>18.3f}")

# ---- composition vs drift decomposition on Q2/Q3 ----
train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
train["sleep_date"] = pd.to_datetime(train["sleep_date"])
r_train = np.array([[rate[t][s] for s in subs] for t in TARGETS])
TAU = 21.0
r_fwd = np.zeros((len(TARGETS), len(subs)))
for ti, t in enumerate(TARGETS):
    for si, s in enumerate(subs):
        d = train[train.subject_id == s]
        last = d["sleep_date"].max()
        w = np.exp(-(last - d["sleep_date"]).dt.days.values / TAU)
        r_fwd[ti, si] = float(np.sum(w * d[t].values) / np.sum(w))
r_hatM = np.array([[r_hat[vix[(s, t)]] for s in subs] for t in TARGETS])

print("\n=== 4. Q2/Q3 public gap = DRIFT (forward-CV) + COMPOSITION residual ===")
print(f"  {'tgt':>4} | {'|gap=rhat-train|':>16} | {'drift explained':>15} | {'composition resid':>17}")
comp_resid = {}
for ti, t in enumerate(TARGETS):
    gap = r_hatM[ti] - r_train[ti]
    drift = r_fwd[ti] - r_train[ti]
    # regress gap on drift (per subject), residual = composition part
    if drift.std() > 1e-9:
        beta = np.dot(gap, drift) / np.dot(drift, drift)
    else:
        beta = 0.0
    resid = gap - beta * drift
    comp_resid[t] = resid
    expl = 1 - np.var(resid) / (np.var(gap) + 1e-12)
    print(f"  {t:>4} | {np.mean(np.abs(gap)):>16.4f} | {expl:>15.2f} | {np.std(resid):>17.4f}")

print("\n  (only Q2/Q3 gap is measurement-informed. If composition residual is ~0, the public")
print("   gap is pure drift -> no shared composition factor to transfer to S.)")
print("\n=== 5. Does a shared composition factor exist across Q2,Q3? (corr of comp-residuals) ===")
c = np.corrcoef(comp_resid["Q2"], comp_resid["Q3"])[0, 1]
print(f"  corr(comp_resid Q2, comp_resid Q3) = {c:+.3f}")
print("  high corr => a real per-subject public-composition factor (transferable to S rows).")
print("  ~0 => composition not identified; only drift is, and drift does not transfer to stable S.")
