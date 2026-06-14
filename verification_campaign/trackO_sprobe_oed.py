#!/usr/bin/env python3
"""TRACK O — S-block OED ceiling + multi-probe sequencing value.

Question: what is the MAX achievable gain by calibrating the S-block to its true public rate,
and how many probe slots does it take to realize it safely?

Bayesian OED using the polytope posterior S (over true public per-subject rates) as BOTH the
prior and the ground-truth generator:
  - For each posterior draw r_true, the infinite-probe move = calibrate FS S-level to r_true.
    Gain G(r_true) = realized S logloss change at r_true. E[G] = the infinite-probe CEILING.
  - k-probe value: probes measure the top-k eigen-directions of the S-rate uncertainty. After
    k probes those directions are pinned; the achievable move = projection. Gain vs k.
This tells us: (a) is the S-block worth probing at all (ceiling << 0?), (b) how many slots,
(c) the downside if the calibration target is wrong (worst-case over the posterior).
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

PE.init()
St = PE._S
subs = St["subs"]; vix = St["vix"]; n_var = St["n_var"]; n_rows = St["n_rows"]
Zfs = St["Zfs"]; Ssamp = St["S"]; rate = St["rate"]
TARGETS = PE.TARGETS
SCELLS_T = ["S1", "S2", "S3", "S4"]
Scells = [vix[(s, t)] for t in SCELLS_T for s in subs]


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def calib_shift(t, s, rho):
    """uniform logit shift so mean pred = rho for subject s, target t."""
    z = Zfs[t][s]; cur = sig(z).mean(); rho = min(max(rho, 0.01), 0.99)
    if abs(cur - rho) < 1e-6:
        return 0.0
    try:
        return float(brentq(lambda dl: sig(z + dl).mean() - rho, -8, 8))
    except Exception:
        return 0.0


def s_gain_at(r_true_vec, move_target_vec):
    """gain (full-1750 metric) of an S-move that calibrates each S cell toward
    move_target_vec[cell], EVALUATED at r_true_vec. neg = improvement.
    Uses the assignment-noise-free functional: per cell,
    dloss = -n_s/N*[ r*Dlogit_term ... ] -> reuse PE-style A + coefs*r."""
    A = 0.0; coefs = np.zeros(n_var)
    for t in SCELLS_T:
        for s in subs:
            j = vix[(s, t)]
            rho = move_target_vec[j]
            dd = calib_shift(t, s, rho)
            if dd == 0:
                continue
            z = Zfs[t][s]
            A += float(-np.log((1 - sig(z + dd)) / (1 - sig(z))).sum()) / 1750.0
            coefs[j] = -n_rows[s] * dd / 1750.0
    return A + coefs @ r_true_vec


# ---- 1. infinite-probe ceiling: calibrate each draw to ITS OWN r_true ----
print("=== 1. Infinite-probe S-block ceiling (calibrate S to the true public rate) ===")
# sub-sample posterior draws for speed
idx = np.linspace(0, len(Ssamp) - 1, 60).astype(int)
gains_perfect = []
for k in idx:
    r_true = Ssamp[k]
    g = s_gain_at(r_true, r_true)   # calibrate to truth, eval at truth
    gains_perfect.append(g)
gains_perfect = np.array(gains_perfect)
print(f"  E[gain] = {gains_perfect.mean():+.5f}   (this is the BEST case: perfect S knowledge)")
print(f"  5th/95th pct = {np.percentile(gains_perfect,5):+.5f} / {np.percentile(gains_perfect,95):+.5f}")
print(f"  => if this is ~0, the S-block is dry even with perfect info; if << 0, probing has upside.")

# ---- 2. fixed-target downside: calibrate to the posterior MEAN, eval at each draw (realistic single move) ----
print("\n=== 2. Single calibrated move to posterior-mean S rate: gain distribution over truth ===")
r_mean = Ssamp.mean(0)
gains_fixed = np.array([s_gain_at(Ssamp[k], r_mean) for k in idx])
print(f"  E[gain] = {gains_fixed.mean():+.5f}   worst(5th best=most positive 95th) = {np.percentile(gains_fixed,95):+.5f}")
print(f"  p(improve) = {(gains_fixed<0).mean():.2f}")
print("  (calibrating to the mean is what you'd do with NO probe; downside = if truth != mean)")

# ---- 3. low-rank structure of S-rate uncertainty: how many probes to pin it ----
print("\n=== 3. S-rate uncertainty low-rank structure (how many probe directions needed) ===")
Scov = np.cov(Ssamp[:, Scells].T)
ev = np.linalg.eigvalsh(Scov)[::-1]
ev = ev[ev > 0]
cum = np.cumsum(ev) / np.sum(ev)
print("  eigenvalue share of S-rate posterior covariance (variance to remove by probing):")
print("  " + "  ".join(f"{e:.3f}" for e in (cum[:8])))
n80 = int(np.searchsorted(cum, 0.80)) + 1
n95 = int(np.searchsorted(cum, 0.95)) + 1
print(f"  probes to remove 80% / 95% of S uncertainty: {n80} / {n95}")

# ---- 4. k-probe realized gain: pin top-k eigendirections, calibrate within pinned subspace ----
print("\n=== 4. k-probe sequencing: realized E[gain] after pinning top-k S directions ===")
w, V = np.linalg.eigh(Scov)
order = np.argsort(w)[::-1]
V = V[:, order]   # columns = S-uncertainty eigenvectors (in Scells space)
print(f"  {'#probes':>8} | {'E[realized gain]':>16} | {'p(improve)':>11} | {'95th (downside)':>15}")
for kprobe in [0, 1, 2, 3, 4, 6, len(Scells)]:
    # pinned subspace = top-k eigvecs; the realized move calibrates toward truth projected onto pinned dirs
    Vk = V[:, :kprobe] if kprobe > 0 else np.zeros((len(Scells), 0))
    P = Vk @ Vk.T if kprobe > 0 else np.zeros((len(Scells), len(Scells)))
    gs = []
    for k in idx:
        r_true = Ssamp[k]
        # target rate on S cells = mean + projection of (truth-mean) onto pinned dirs
        delta_s = (r_true[Scells] - r_mean[Scells])
        tgt_s = r_mean[Scells] + P @ delta_s
        tgt = r_mean.copy(); tgt[Scells] = tgt_s
        gs.append(s_gain_at(r_true, tgt))
    gs = np.array(gs)
    print(f"  {kprobe:>8} | {gs.mean():>+16.5f} | {(gs<0).mean():>11.2f} | {np.percentile(gs,95):>+15.5f}")

print("\n  Interpretation: E[realized gain] should rise (more negative) toward the ceiling (panel 1)")
print("  as probes increase. If even the ceiling is small, the S-probe campaign cannot beat 0.5619 much.")
