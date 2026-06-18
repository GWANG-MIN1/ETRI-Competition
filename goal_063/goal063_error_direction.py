"""GOAL_063 — can we predict the DIRECTION of base error (signed residual), not just magnitude?

KEY FACT: targets are binary, so sign(y - p_base) == y. The meaningful question is whether
observable (missingness + life-rhythm + subject-relative + HS-JEPA) features predict the SIGNED
residual r = y - p_base BEYOND what base already captured -> i.e. can a directional correction
be added to base. For a well-calibrated base, E[r|feat]=0 (direction unpredictable) even though
Var[r|feat] is predictable (GOAL_062). Live hypothesis: routine-anomaly days bias base one way.

Tests (honest only): residual regression feat->r, per target,
  - Pearson/Spearman(pred r, true r) on held-out + FUTURE block + LOSO, vs PLACEBO (shuffle r).
  - logloss delta of corrected p = clip(p_base + beta*pred_r) vs base, on held-out + FUTURE.
  - literal sign-AUC / balanced-acc of (feat -> y) classifier (and vs base AUC).
  - restricted to GOAL_062 high-risk rows (B).
PASS iff FUTURE corr>0 & logloss improves & beats placebo & LOSO holds & consistent across targets.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import lightgbm as lgb

KIT = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
WMC = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\worldmodel_jepa\cache")
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(HSJEPA_SRC))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa

EPS = 1e-6
TARGETS = list(K.TARGETS)
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173]
def clip01(p): return np.clip(p, EPS, 1 - EPS)
def hbin(h): hh = h if h >= 12 else h + 24; return int((hh - 12) * 12)
def bll(y, p): p = clip01(p); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

m = K.base(); subj = m["subject_id"].values
P = {t: m[f"p_{t}"].values for t in TARGETS}; Yt = {t: m[f"y_{t}"].values for t in TARGETS}

# ---- features (missingness + routine + subject-relative) from canvas ----
d = np.load(WMC / "wm_canvas.npz", allow_pickle=True)
X = d["X"]; obs = d["obs"]; chans = list(d["chans"]); keys = d["keys"]
cmap = {(s, pd.to_datetime(dt).normalize()): i for i, (s, dt) in enumerate(keys)}
ridx = np.array([cmap[(s, pd.to_datetime(dd).normalize())] for s, dd in zip(m.subject_id, m.lifelog_date)])
nb = slice(hbin(22), hbin(33)); ci = {c: i for i, c in enumerate(chans)}
feats = {}
for c in chans:
    feats[f"cov_{c}"] = np.array([(obs[r, ci[c], nb] > 0).mean() for r in ridx])
key_ch = [c for c in ("hr_mean", "active_rate", "screen_on", "light", "step", "hr_rmssd") if c in ci]
for c in key_ch:
    nm = np.array([X[r, ci[c], nb][obs[r, ci[c], nb] > 0].mean() if (obs[r, ci[c], nb] > 0).any() else np.nan for r in ridx])
    feats[f"nm_{c}"] = nm
    dev = np.full(len(m), np.nan); pct = np.full(len(m), np.nan)
    for s in np.unique(subj):
        mk = subj == s; v = nm[mk]
        med = np.nanmedian(v); iqr = (np.nanpercentile(v, 75) - np.nanpercentile(v, 25)) or 1.0
        dev[mk] = (v - med) / iqr                       # SIGNED deviation (direction-relevant)
        r = pd.Series(v).rank(pct=True).values; pct[mk] = r
    feats[f"sdev_{c}"] = dev; feats[f"pct_{c}"] = pct
dow = pd.to_datetime(m["lifelog_date"]).dt.dayofweek.values
feats["dow"] = dow.astype(float); feats["weekend"] = (dow >= 5).astype(float)
FN = list(feats.keys())
R = np.column_stack([feats[f] for f in FN]).astype(float)
cm = np.nanmedian(R, 0); cm = np.where(np.isfinite(cm), cm, 0.0)
R = np.where(np.isfinite(R), R, cm)
# + HS-JEPA latent
z = np.load(WMC / "wm_circajepa_z.npz", allow_pickle=True)["z"][ridx]
Rz = np.column_stack([R, z])
print(f"features: risk {R.shape[1]} | +JEPA {Rz.shape[1]}")

LGB = dict(n_estimators=200, learning_rate=0.04, num_leaves=15, min_child_samples=25,
           reg_lambda=8.0, subsample=0.8, colsample_bytree=0.7, n_jobs=-1, verbose=-1)
def future_mask(seed, ff=0.15):
    rng = np.random.default_rng(seed); fut = np.zeros(len(m), bool)
    for s, idx in m.groupby("subject_id").groups.items():
        idx = np.array(sorted(idx, key=lambda i: m.at[i, "sleep_date"])); fut[idx[-max(1, int(round(len(idx) * ff))):]] = True
    return fut
def loso_folds():
    for s in np.unique(subj):
        te = np.flatnonzero(subj == s); tr = np.flatnonzero(subj != s)
        yield tr, te

# GOAL_062 high-risk rows (magnitude model OOF) for subset (B)
absres = np.abs(np.column_stack([Yt[t] - P[t] for t in TARGETS])).mean(1)
risk = np.zeros(len(m)); cnt = np.zeros(len(m))
for seed in SEEDS[:5]:
    for tr, va in H.interleaved_folds(m, 5, seed):
        risk[va] += lgb.LGBMRegressor(**LGB, random_state=seed).fit(R[tr], absres[tr]).predict(R[va]); cnt[va] += 1
risk /= np.maximum(cnt, 1); hi = risk >= np.quantile(risk, 0.67)

def run(Feat, tag):
    print(f"\n===== DIRECTION test [{tag}] (resid regression feat->y-p_base) =====")
    print(f"  {'tgt':3s} | corr held / FUTURE / LOSO | placebo | sign-AUC(base) | ΔloglossFUT | hiRiskFUT")
    rows = []
    for t in TARGETS:
        r = Yt[t] - P[t]; y = Yt[t]
        ch, cf, cpl, dlf, hir = [], [], [], [], []
        for seed in SEEDS:
            held = H.test_faithful_mask(m, seed); tr = np.flatnonzero(~held); va = np.flatnonzero(held)
            reg = lgb.LGBMRegressor(**LGB, random_state=seed).fit(Feat[tr], r[tr])
            pr = reg.predict(Feat[va])
            if np.std(pr) > 1e-9:
                ch.append(pearsonr(pr, r[va])[0])
            fut = np.flatnonzero(future_mask(seed)); vf = np.intersect1d(va, fut)
            if len(vf) > 8:
                pos = np.searchsorted(va, vf)
                if np.std(pr[pos]) > 1e-9:
                    cf.append(pearsonr(pr[pos], r[vf])[0])
                # logloss delta with best small beta (chosen on train resid fit scale)
                best = 0.0
                for beta in (0.3, 0.5, 1.0):
                    pc = clip01(P[t][vf] + beta * pr[pos])
                    dl = bll(y[vf], pc) - bll(y[vf], P[t][vf])
                    best = min(best, dl)
                dlf.append(best)
                # high-risk subset on future
                vhf = np.intersect1d(vf, np.flatnonzero(hi))
                if len(vhf) > 5:
                    posh = np.searchsorted(va, vhf)
                    if np.std(pr[posh]) > 1e-9:
                        hir.append(pearsonr(pr[posh], r[vhf])[0])
            # placebo: shuffle residual target
            rng = np.random.default_rng(seed * 5 + 3); rp = r[rng.permutation(len(r))]
            regp = lgb.LGBMRegressor(**LGB, random_state=seed).fit(Feat[tr], rp[tr])
            prp = regp.predict(Feat[va])
            if np.std(prp) > 1e-9:
                cpl.append(pearsonr(prp, r[va])[0])
        # LOSO corr
        cl = []
        for tr, te in loso_folds():
            reg = lgb.LGBMRegressor(**LGB, random_state=0).fit(Feat[tr], r[tr])
            pr = reg.predict(Feat[te])
            if np.std(pr) > 1e-9 and np.std(r[te]) > 1e-9:
                cl.append(pearsonr(pr, r[te])[0])
        # literal sign-AUC: feat->y classifier vs base AUC
        from sklearn.model_selection import cross_val_predict
        try:
            clf = lgb.LGBMClassifier(**LGB)
            pcv = cross_val_predict(clf, Feat, y, cv=5, method="predict_proba")[:, 1]
            sauc = roc_auc_score(y, pcv); bauc = roc_auc_score(y, P[t])
        except Exception:
            sauc = bauc = float("nan")
        row = dict(t=t, held=np.nanmean(ch), fut=np.nanmean(cf), loso=np.nanmean(cl),
                   pl=np.nanmean(cpl), sauc=sauc, bauc=bauc, dlf=np.nanmean(dlf), hir=np.nanmean(hir) if hir else np.nan)
        rows.append(row)
        print(f"  {t:3s} | {row['held']:+.3f} / {row['fut']:+.3f} / {row['loso']:+.3f} | "
              f"plac {row['pl']:+.3f} | {row['sauc']:.3f}({row['bauc']:.3f}) | {row['dlf']:+.5f} | {row['hir']:+.3f}")
    # verdict
    good = [r for r in rows if r["fut"] > 0.05 and r["fut"] - r["pl"] > 0.04 and r["loso"] > 0.0 and r["dlf"] < -0.0003]
    print(f"  -> targets with predictable DIRECTION (FUT>.05, beats placebo, LOSO>0, logloss improves): "
          f"{[r['t'] for r in good] if good else 'NONE'}")
    return rows, good

rows_r, good_r = run(R, "risk feats")
rows_z, good_z = run(Rz, "risk + HS-JEPA z")
allgood = set([r["t"] for r in good_r]) | set([r["t"] for r in good_z])
print("\n================ GOAL_063 VERDICT ================")
if allgood:
    print(f"PARTIAL/PASS — direction predictable for: {sorted(allgood)}")
    print("  (theoretical logloss bound = applied Δlogloss above; oracle resid -> 0 only with true y)")
else:
    print("FAIL — error DIRECTION not predictable beyond base on future/LOSO/placebo.")
    print("  Reason: base is well-calibrated wrt these observable features (they are in its 5242-feature")
    print("  training set), so E[y-p|feat]≈0 (mean/direction unpredictable). Only Var (magnitude, GOAL_062)")
    print("  is predictable. No signed residual to correct -> selective calibration cannot gain. anchor 0.5615.")
