"""GOAL_061 — HS-JEPA cohort pooling (representation-similarity per-subject LEVEL pooling).

Thesis: transferable class = per-subject LEVEL; n=10 makes per-subject rate estimates noisy.
base treats subjects via per-row features but does NOT pool subjects by representation. Use the
frozen HS-JEPA subject embedding (mean nightly z) -> subject-subject cosine similarity ->
partial-pool each subject's per-target rate toward JEPA-similar neighbors -> apply as a
per-subject logit level shift on base.

Decisive vetting (must beat ALL, esp. on the FUTURE block, else FAIL):
  vs base, vs own-train-rate re-level, vs shrink-to-GLOBAL (the 0610 control), vs PLACEBO
  (shuffled subject similarity). Plus authoritative transfer gate (PASS_LEVEL required).
Level-class only; no per-row, no leakage (embedding=sensors; rates=train labels only).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

KIT = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
WMC = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\worldmodel_jepa\cache")
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(HSJEPA_SRC))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa
from gate_transfer_vet import vet_candidate  # noqa

EPS = 1e-6
TARGETS = list(K.TARGETS)
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173]


def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))


m = K.base()
# align frozen HS-JEPA z to the 450 rows
z700 = np.load(WMC / "wm_circajepa_z.npz", allow_pickle=True)["z"]
keys700 = np.load(WMC / "wm_drift_feature.npz", allow_pickle=True)["keys"]
cmap = {(s, pd.to_datetime(d).normalize()): i for i, (s, d) in enumerate(keys700)}
ridx = np.array([cmap[(s, pd.to_datetime(d).normalize())] for s, d in zip(m.subject_id, m.lifelog_date)])
Z = z700[ridx]                                   # (450,96)
subj = m["subject_id"].values
subs = sorted(np.unique(subj))

# subject embeddings (mean nightly z, label-free) + cosine similarity
emb = np.array([Z[subj == s].mean(0) for s in subs])
embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
SIM = embn @ embn.T                              # (10,10) cosine
np.fill_diagonal(SIM, -np.inf)                   # exclude self
print("subjects:", subs)
print("similarity row example (id of nearest neighbor per subject):")
for i, s in enumerate(subs):
    j = np.argmax(SIM[i]); print(f"  {s} -> nearest {subs[j]} (cos {SIM[i][j]:.3f})")
offdiag = SIM[np.isfinite(SIM)]
print(f"similarity spread (off-diag): mean {offdiag.mean():.3f} std {offdiag.std():.3f}  "
      f"(near-uniform => pooling ~ shrink-to-global)")


def neighbor_weights(tau):
    W = np.zeros_like(SIM)
    for i in range(len(subs)):
        s = SIM[i].copy()
        w = np.exp((s - np.nanmax(s[np.isfinite(s)])) / tau)
        w[~np.isfinite(s)] = 0.0
        W[i] = w / (w.sum() + 1e-12)
    return W                                     # rows sum to 1, self=0


def pooled_rate(rate_by_s, W, alpha):
    r = np.array([rate_by_s[s] for s in subs])
    neigh = W @ r
    return {s: (1 - alpha) * r[i] + alpha * neigh[i] for i, s in enumerate(subs)}


# ---- forward-CV: does pooled per-subject rate predict FUTURE rows better than base/own/global/placebo? ----
def split_future(m, seed, ff=0.15):
    rng = np.random.default_rng(seed); fut = np.zeros(len(m), bool)
    for s, idx in m.groupby("subject_id").groups.items():
        idx = np.array(sorted(idx, key=lambda i: m.at[i, "sleep_date"]))
        nf = max(1, int(round(len(idx) * ff))); fut[idx[-nf:]] = True
    return np.flatnonzero(fut)


def relevel(base_p, shift_by_s):
    z = logit(base_p).copy()
    for i in range(len(z)):
        z[i] += shift_by_s[subj[i]]
    return sig(z)


def eval_method(t, alpha, tau, beta, placebo_seed=None):
    """mean FUTURE-block logloss delta vs base for: pooled / own / global / placebo."""
    y = m[f"y_{t}"].values; base = m[f"p_{t}"].values
    res = {k: [] for k in ("own", "global", "pooled", "placebo")}
    base_fut = []
    for seed in SEEDS:
        fut = split_future(m, seed); tr = np.setdiff1d(np.arange(len(m)), fut)
        rate = {s: float(y[tr][subj[tr] == s].mean()) if (subj[tr] == s).any() else float(y[tr].mean()) for s in subs}
        gl = float(y[tr].mean())
        b_s = {s: float(base[tr][subj[tr] == s].mean()) for s in subs}  # base implied level
        W = neighbor_weights(tau)
        if placebo_seed is not None:
            rng = np.random.default_rng(placebo_seed + seed)
            perm = rng.permutation(len(subs)); Wp = W[perm][:, perm]
        pooled = pooled_rate(rate, W, alpha)
        shifts = {
            "own": {s: beta * (logit(rate[s]) - logit(b_s[s])) for s in subs},
            "global": {s: beta * (logit(gl) - logit(b_s[s])) for s in subs},
            "pooled": {s: beta * (logit(pooled[s]) - logit(b_s[s])) for s in subs},
        }
        if placebo_seed is not None:
            pl = pooled_rate(rate, Wp, alpha)
            shifts["placebo"] = {s: beta * (logit(pl[s]) - logit(b_s[s])) for s in subs}
        ll_base = H.bll(y[fut], base[fut]); base_fut.append(ll_base)
        for k, sh in shifts.items():
            p = relevel(base, sh)
            res[k].append(H.bll(y[fut], p[fut]) - ll_base)
    out = {k: (float(np.mean(v)), float(np.mean(np.array(v) < 0))) for k, v in res.items() if v}
    return out


print("\n=== forward-CV (FUTURE block) logloss Δ vs base  (neg=better) ===")
print("  best per target over alpha/tau/beta grid; pooled must beat own/global/placebo")
ALPHAS = [0.3, 0.5, 0.7]; TAUS = [0.05, 0.15]; BETAS = [0.5, 1.0]
summary = {}
for t in TARGETS:
    best = None
    for a in ALPHAS:
        for tau in TAUS:
            for b in BETAS:
                r = eval_method(t, a, tau, b, placebo_seed=999)
                key = (a, tau, b)
                if best is None or r["pooled"][0] < best[1]["pooled"][0]:
                    best = (key, r)
    (a, tau, b), r = best
    summary[t] = r
    win = (r["pooled"][0] < -0.0005 and r["pooled"][0] < r["own"][0] - 1e-6
           and r["pooled"][0] < r["global"][0] - 1e-6 and r["pooled"][0] < r["placebo"][0] - 1e-6)
    flag = "  <<< POOLED WINS" if win else ""
    print(f"  {t} (a{a},tau{tau},b{b}): pooled {r['pooled'][0]:+.5f}(sg{r['pooled'][1]:.2f}) | "
          f"own {r['own'][0]:+.5f} | global {r['global'][0]:+.5f} | placebo {r['placebo'][0]:+.5f}{flag}")

# ---- authoritative transfer gate on the full pooled candidate (best alpha/tau/beta = a0.5,tau0.15,b1.0) ----
print("\n=== authoritative transfer gate (full pooled candidate, deploy: rates on ALL train) ===")
W = neighbor_weights(0.15)
cand = m[H.KEYS].copy()
for t in TARGETS:
    cand[t] = m[f"p_{t}"].values
for t in TARGETS:
    y = m[f"y_{t}"].values; base = m[f"p_{t}"].values
    rate = {s: float(y[subj == s].mean()) for s in subs}
    b_s = {s: float(base[subj == s].mean()) for s in subs}
    pooled = pooled_rate(rate, W, 0.5)
    sh = {s: 1.0 * (logit(pooled[s]) - logit(b_s[s])) for s in subs}
    cand[t] = relevel(base, sh)
r = vet_candidate(cand, label="G61_cohort_pooling", verbose=False)
verds = {t: r["targets"][t]["verdict"] for t in TARGETS}
pass_level = [t for t in TARGETS if verds[t] == "PASS_LEVEL"]
print("  verdicts:", verds)
print(f"  PASS_LEVEL: {pass_level if pass_level else 'NONE'}")

wins = [t for t in TARGETS if (summary[t]["pooled"][0] < -0.0005
        and summary[t]["pooled"][0] < summary[t]["own"][0] - 1e-6
        and summary[t]["pooled"][0] < summary[t]["global"][0] - 1e-6
        and summary[t]["pooled"][0] < summary[t]["placebo"][0] - 1e-6)]
print(f"\nGOAL_061 VERDICT: {'PARTIAL/PASS' if (wins and pass_level) else 'FAIL'}")
print(f"  forward-CV pooled-wins targets: {wins if wins else 'NONE'}")
print(f"  gate PASS_LEVEL targets: {pass_level if pass_level else 'NONE'}")
if not (wins and pass_level):
    print("  -> cohort pooling does not beat own/global/placebo on future AND/OR gate not PASS_LEVEL. anchor 0.5615 유지.")
