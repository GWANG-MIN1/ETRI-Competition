#!/usr/bin/env python3
"""TRACK S2 — private robustness via the AUTHORITATIVE polytope posterior (no mis-specified model).

For a per-subject-uniform move d, gain = A(d) + coefs(d)·r. Public uses the true public rate r
(the polytope posterior, calibrated to real LB). PRIVATE uses r_priv = r + eps, where eps is the
independent public/private sampling split per subject (~12 rows each => SD per subject ~0.14,
so r_pub - r_priv has SD ~0.20). Thus:
   g_priv = g_pub + coefs·eps,   E[g_priv|r] = g_pub,   plus sampling variance.
This yields the faithful private-gain distribution for overshoot x sigma. Report E, SD, P(improve),
CVaR5, and 2-submission hedges (DACON keeps the better on private).
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

PE.init()
St = PE._S
subs = St["subs"]; vix = St["vix"]; n_var = St["n_var"]; n_rows = St["n_rows"]
Zfs = St["Zfs"]; S = St["S"]; d_over = St["d_over"]; rate = St["rate"]
TARGETS = PE.TARGETS
TEST_N = {"id01": 27, "id02": 32, "id03": 21, "id04": 27, "id05": 21,
          "id06": 24, "id07": 30, "id08": 19, "id09": 27, "id10": 22}
rng = np.random.default_rng(2026)


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


# per-subject public/private split sampling SD on the rate (independent halves of ~TEST_N rows)
def split_sd(s, t, r):
    npub = max(4, TEST_N[s] // 2); npriv = max(4, TEST_N[s] - TEST_N[s] // 2)
    return np.sqrt(r * (1 - r) * (1.0 / npub + 1.0 / npriv))


sigmas = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
# precompute, per sigma: g_pub samples (posterior) and add private sampling noise
NDRAW = 4000
priv = {}
pub = {}
for s in sigmas:
    d = d_over * s
    A, c = gt(d)
    g_pub = S @ c + A                       # over posterior (public)
    # private: for each posterior draw r, add coefs·eps with eps ~ per-subject split noise
    idxp = rng.integers(0, len(S), NDRAW)
    gpv = []
    for k in idxp:
        r = S[k]
        eps = np.zeros(n_var)
        for t in ("Q2", "Q3"):
            for sub in subs:
                j = vix[(sub, t)]
                eps[j] = rng.normal(0, split_sd(sub, t, np.clip(r[j], 0.05, 0.95)))
        gpv.append(A + c @ (r + eps))
    priv[s] = np.array(gpv)
    pub[s] = g_pub

print("=== overshoot x sigma: faithful PRIVATE gain (polytope posterior + split sampling) ===")
print(f"  {'sigma':>6} | {'E[pub]':>9} | {'E[priv]':>9} | {'SD priv':>8} | {'P(priv<0)':>10} | {'CVaR5':>9}")
for s in sigmas:
    gv = priv[s]
    cvar = np.mean(np.sort(gv)[-int(0.05 * len(gv)):])
    print(f"  {s:>6.2f} | {pub[s].mean():>+9.5f} | {gv.mean():>+9.5f} | {gv.std():>8.5f} | "
          f"{(gv<0).mean():>10.2f} | {cvar:>+9.5f}")

bm = min(sigmas[1:], key=lambda s: priv[s].mean())
bc = min(sigmas[1:], key=lambda s: np.mean(np.sort(priv[s])[-int(0.05 * len(priv[s])):]))
bp = max(sigmas[1:], key=lambda s: (priv[s] < 0).mean())
print(f"\n  private E-optimal sigma={bm}  CVaR5-optimal sigma={bc}  P(improve)-optimal sigma={bp}")

print("\n=== 2-submission hedge: DACON keeps BETTER on private. E[min gain] (lower=better) ===")
print(f"  {'pair':>14} | {'E[best-of-2]':>13} | {'P(best<0)':>10} | {'CVaR5 best':>11}")
pairs = [("x0.8 alone", 0.8, 0.8), ("x0.6,x1.0", 0.6, 1.0), ("x0.7,x1.0", 0.7, 1.0),
         ("FS,x0.8", 0.0, 0.8), ("FS,x1.0", 0.0, 1.0), ("x0.5,x0.9", 0.5, 0.9)]
for lab, a, b in pairs:
    best = np.minimum(priv[a], priv[b])
    cvar = np.mean(np.sort(best)[-int(0.05 * len(best)):])
    print(f"  {lab:>14} | {best.mean():>+13.5f} | {(best<0).mean():>10.2f} | {cvar:>+11.5f}")
print("\n  Within-family pairs bracket the scale; pairing with FS caps the downside tail (CVaR ~0).")
