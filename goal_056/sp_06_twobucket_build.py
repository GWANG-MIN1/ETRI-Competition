"""sp_06: Two-bucket (future vs interleaved) overshoot — BUILD + DUAL ARBITER.

sp_05 showed (forward-CV) that interleaved test rows want a SMALLER recency overshoot
than future rows. The deployed best (0.5615) applies a UNIFORM sigma=0.8 to all rows,
so it over-shoots the 62.4% interleaved majority. The bucket label (row date > subject's
max train date) is OBSERVED (zero estimation noise) => stays in the transferable level
class, Q2/Q3 only. This script turns that into actual TEST submissions and scores each
candidate on TWO independent arbiters:

  ARBITER 1 (PE / polytope_eval) = LB-CALIBRATED. Posterior constrained by real public-LB
    observations; certified uniform 0.8 as robust-optimal. Scores the per-subject AVERAGE
    shift => "does this regress the validated per-subject level?" (PE can't see within-
    subject future/interleaved label differences, so it is a CONSERVATIVE safety floor).
  ARBITER 2 (forward-CV on train, sp_05 style) = rewards the within-subject redistribution
    that PE is blind to. Gain vs the deployed uniform-0.8, blended 37.6/62.4.

A defensible candidate is one that forward-CV likes (gain<0) AND PE does not punish
(post-mean >= deployed-0.8's, i.e. per-subject level preserved). NO LB fitting. Verifies
the deployed best is exactly reproducible before building anything.
"""
from __future__ import annotations
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FHC = HERE.parent  # final_hsjepa_candidates
KIT = FHC / "outputs"
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(FHC)); sys.path.insert(0, str(KIT)); sys.path.insert(0, str(HSJEPA_SRC))

import polytope_eval as PE  # noqa: E402
import goal054_kit as K  # noqa: E402
import hsjepa_core as H  # noqa: E402

EPS = 1e-6
KEYS = PE.KEYS
TARGETS = PE.TARGETS
ETRI = PE.ETRI
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173, 197, 211]


def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------- setup
print("init PE (LB-calibrated polytope) ...")
PE.init()
S = PE._S
subs = S["subs"]
vix = S["vix"]
d_over = S["d_over"]  # per-(subject,target) LOGIT step; nonzero only for Q2/Q3 (sigma=1.0 anchor)

fs = pd.read_csv(ETRI / PE.FS_FILE).sort_values(KEYS).reset_index(drop=True)
sample = pd.read_csv(ETRI / "data" / "ch2026_submission_sample.csv").sort_values(KEYS).reset_index(drop=True)
fsub = fs["subject_id"].values
fs_dates = pd.to_datetime(fs["sleep_date"]).values

train = pd.read_csv(ETRI / "data" / "ch2026_metrics_train.csv")
tmax_ser = pd.to_datetime(train.groupby("subject_id")["sleep_date"].max())
tmax = {s: np.datetime64(v) for s, v in tmax_ser.items()}

is_future = np.array([fs_dates[i] > tmax[fsub[i]] for i in range(len(fs))])
print(f"TEST rows={len(fs)}  future={int(is_future.sum())} ({is_future.mean():.3f})  "
      f"interleaved={int((~is_future).sum())} ({1 - is_future.mean():.3f})  "
      f"[doc structure: ~0.376 future / 0.624 interleaved]")


def row_step(t):
    return np.array([d_over[vix[(s, t)]] for s in fsub])


def build(sig_int, sig_fut):
    """Two-bucket test submission: interleaved rows get sig_int*step, future rows sig_fut*step (Q2/Q3)."""
    out = fs[KEYS].copy()
    for t in TARGETS:
        z = logit(fs[t].values).copy()
        if t in ("Q2", "Q3"):
            sg = np.where(is_future, sig_fut, sig_int)
            z = z + sg * row_step(t)
        out[t] = sig(z)
    out = out.set_index(KEYS).loc[sample.set_index(KEYS).index].reset_index()
    return out


# ----------------------------------------------------------- VERIFY reconstruction
ref = pd.read_csv(ETRI / "submission_overshoot_x0p8_330ef1a1_uploadsafe.csv")
dep = build(0.8, 0.8)
cmp = dep[KEYS + TARGETS].merge(ref[KEYS + TARGETS], on=KEYS, suffixes=("_new", "_ref"))
maxdiff = max(np.abs(cmp[f"{t}_new"].to_numpy(float) - cmp[f"{t}_ref"].to_numpy(float)).max() for t in TARGETS)
print(f"\nRECONSTRUCT deployed best  build(0.8,0.8) vs 0.5615 file: max|diff| = {maxdiff:.2e}")
assert maxdiff < 1e-9, "RECONSTRUCTION MISMATCH — base/step/space wrong, aborting"
print("  OK -> base, step, logit-space, tau all match the deployed 0.5615 file.\n")


# ------------------------------------------------- ARBITER 1: PE (LB-calibrated)
def pe_delta_for(sig_int, sig_fut):
    """Per-subject AVERAGE shift over that subject's test rows (PE granularity)."""
    delta = {}
    for s in subs:
        msk = fsub == s
        nf = int(is_future[msk].sum()); ni = int((~is_future[msk]).sum()); nt = int(msk.sum())
        avg = (nf * sig_fut + ni * sig_int) / nt
        for t in ("Q2", "Q3"):
            delta[(s, t)] = avg * d_over[vix[(s, t)]]
    return delta


def pe_score(sig_int, sig_fut):
    return PE.eval_delta(pe_delta_for(sig_int, sig_fut), settings18=True, verbose=False)


# ----------------------------------------- ARBITER 2: forward-CV on train (sp_05)
def split_future_inter(m, seed, future_frac=0.15, inter_frac=0.25):
    rng = np.random.default_rng(seed)
    fut = np.zeros(len(m), bool); inter = np.zeros(len(m), bool)
    for s, idx in m.groupby("subject_id").groups.items():
        idx = np.array(sorted(idx, key=lambda i: m.at[i, "sleep_date"]))
        n = len(idx)
        n_fut = max(1, int(round(n * future_frac)))
        f = idx[-n_fut:]; rest = idx[:-n_fut]
        n_int = max(1, int(round(n * inter_frac)))
        it = rng.choice(rest, size=min(n_int, len(rest)), replace=False)
        fut[f] = True; inter[it] = True
    return np.flatnonzero(fut), np.flatnonzero(inter)


def recency_step(m, tr, t, tau=21.0):
    trdf = m.iloc[tr]; step = {}
    for s, idx in trdf.groupby("subject_id").groups.items():
        d = trdf.loc[idx]; last = d["sleep_date"].max()
        w = np.exp(-(last - d["sleep_date"]).dt.days.values / tau)
        rate = float(np.sum(w * d[f"y_{t}"].values) / np.sum(w))
        smean = float(d[f"y_{t}"].values.mean())
        step[s] = logit(np.clip(rate, EPS, 1 - EPS)) - logit(np.clip(smean, EPS, 1 - EPS))
    return step


_M = K.base()


def fwdcv_gain(sig_int, sig_fut):
    """Per-target forward-CV logloss gain vs uniform-0.8 (blended 37.6/62.4). gain<0 = better."""
    res = {"Q2": [], "Q3": []}
    for t in ("Q2", "Q3"):
        for seed in SEEDS:
            fidx, iidx = split_future_inter(_M, seed)
            tr = np.setdiff1d(np.arange(len(_M)), np.concatenate([fidx, iidx]))
            step = recency_step(_M, tr, t)
            subj = _M["subject_id"].values; y = _M[f"y_{t}"].values; z0 = logit(_M[f"p_{t}"].values)
            st = np.array([step.get(s, 0.0) for s in subj])
            p08 = sig(z0 + 0.8 * st)
            ll08 = 0.376 * H.bll(y[fidx], p08[fidx]) + 0.624 * H.bll(y[iidx], p08[iidx])
            pz = z0.copy()
            pz[fidx] = z0[fidx] + sig_fut * st[fidx]
            pz[iidx] = z0[iidx] + sig_int * st[iidx]
            lltb = 0.376 * H.bll(y[fidx], sig(pz[fidx])) + 0.624 * H.bll(y[iidx], sig(pz[iidx]))
            res[t].append(lltb - ll08)
    out = {}
    for t in ("Q2", "Q3"):
        a = np.array(res[t]); out[t] = (float(a.mean()), float(np.mean(a < 0)))
    return out


# --------------------------------------------------------------------- references
print("PE reference points (post-mean = gain vs FS base; more negative = better):")
for lbl, si, sf in [("anchor x1.0  ", 1.0, 1.0), ("deployed x0.8", 0.8, 0.8)]:
    r = pe_score(si, sf)
    print(f"  {lbl}: post-mean {r['post_mean']:+.5f}  worst-loose {r['worst_loose']:+.5f}  "
          f"fav18 {r['fav18']:.2f}  P(<0) {r['p_improve']:.2f}")
dep08 = pe_score(0.8, 0.8)

# --------------------------------------------------------------------- the sweep
print("\n=== TWO-BUCKET SWEEP ===")
print("cfg(int,fut) | PE post-mean (Δ vs dep0.8) | PE worst | fav18 |"
      " fwdCV ΔQ2(sgn) ΔQ3(sgn) | 7t-mean Δ")
rows = []
for sig_int in [0.0, 0.2, 0.4, 0.6]:
    for sig_fut in [0.8, 1.0, 1.2, 1.4]:
        pr = pe_score(sig_int, sig_fut)
        fc = fwdcv_gain(sig_int, sig_fut)
        seven = (fc["Q2"][0] + fc["Q3"][0]) / 7.0
        rows.append(dict(si=sig_int, sf=sig_fut, pe=pr["post_mean"], pe_w=pr["worst_loose"],
                         fav=pr["fav18"], q2=fc["Q2"][0], q2s=fc["Q2"][1], q3=fc["Q3"][0],
                         q3s=fc["Q3"][1], seven=seven))
        print(f"  ({sig_int:.1f},{sig_fut:.1f})   | {pr['post_mean']:+.5f} "
              f"({pr['post_mean'] - dep08['post_mean']:+.5f}) | {pr['worst_loose']:+.5f} | {pr['fav18']:.2f} |"
              f" {fc['Q2'][0]:+.5f}({fc['Q2'][1]:.2f}) {fc['Q3'][0]:+.5f}({fc['Q3'][1]:.2f}) | {seven:+.6f}")

print("\nDefensible = fwdCV 7t-mean Δ < 0  AND  PE post-mean >= deployed-0.8 (level not regressed):")
for r in rows:
    ok = (r["seven"] < 0) and (r["pe"] <= dep08["post_mean"] + 1e-6)
    if ok:
        print(f"  ** ({r['si']:.1f},{r['sf']:.1f})  PEΔ {r['pe'] - dep08['post_mean']:+.5f}  "
              f"fwdCV 7t {r['seven']:+.6f}  worst {r['pe_w']:+.5f}")


# --------------------------------------------------------------------- SAVE chosen
def save(sig_int, sig_fut, tag):
    out = build(sig_int, sig_fut)
    out = out[sample.columns.tolist()]
    prob = out[TARGETS].to_numpy(float)
    base = ref[TARGETS].to_numpy(float)  # deployed 0.5615 as reference for "what changed"
    base_aligned = ref.set_index(KEYS).loc[out.set_index(KEYS).index][TARGETS].to_numpy(float)
    dd = np.abs(prob - base_aligned)
    digest = hashlib.sha1(np.round(prob, 12).tobytes()).hexdigest()[:8]
    checks = {
        "shape (250,10)": out.shape == (250, 10),
        "keys==sample": out[KEYS].reset_index(drop=True).equals(sample[KEYS].reset_index(drop=True)),
        "null==0": int(out[TARGETS].isnull().sum().sum()) == 0,
        "strictly in (0,1)": bool(prob.min() > 0 and prob.max() < 1),
        "only Q2/Q3 changed vs deployed": [t for i, t in enumerate(TARGETS) if dd[:, i].max() > 1e-12] == ["Q2", "Q3"],
    }
    assert all(checks.values()), f"upload-safety FAILED for {tag}: {checks}"
    fn = ETRI / f"submission_twobucket_{tag}_{digest}_uploadsafe.csv"
    out.to_csv(fn, index=False)
    print(f"\nSAVED {fn.name}")
    print(f"  upload-safety: ALL OK  | hash {digest}")
    print(f"  vs deployed 0.5615: changed cells {int((dd > 1e-12).sum())} (interleaved Q2/Q3), "
          f"mean|Δ| {dd[dd > 1e-12].mean():.4f}  max|Δ| {dd.max():.4f}")
    print(f"  Q2 mean {ref['Q2'].mean():.4f}->{out['Q2'].mean():.4f}  "
          f"Q3 mean {ref['Q3'].mean():.4f}->{out['Q3'].mean():.4f}")


print("\n=== BUILD recommended candidates ===")
save(0.6, 1.0, "int0p6_fut1p0")  # private-safe primary  (PE -0.00048 vs dep, worst-loose -0.00048)
save(0.4, 1.2, "int0p4_fut1p2")  # aggressive option     (PE -0.00082 vs dep, worst-loose +0.00089)
print("\nDONE")
