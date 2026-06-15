#!/usr/bin/env python3
"""GOAL — per-row breakthrough probe: cross-target + reconstruction stacking, transfer-vetted.

If the team modeled the 7 targets (largely) independently, then for each target y_t a model on
[all 7 OOF preds + reconstructed sleep features] could beat p_t alone — esp. Q1/Q2/Q3 (perceived
quality/fatigue/stress) which are DOWNSTREAM of objective sleep S1-S4. Test under interleaved CV
(transfer-honest), report logloss delta, sign-stability, and a level/residual split of the gain.
Only a gain that clears the transfer floor AND is sign-stable counts as a real per-row lever.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
CORE = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(CORE))
import hsjepa_core as H  # noqa: E402

EPS = 1e-6
TARGETS = H.TARGETS
m = H.load()
recon = pd.read_parquet(HERE / "sleep_recon_train.parquet")
recon["sleep_date"] = pd.to_datetime(recon["sleep_date"]); recon["lifelog_date"] = pd.to_datetime(recon["lifelog_date"])
rcols = [c for c in ["tst", "tib", "se", "sol", "waso", "onset_h"] if c in recon.columns]
mm = m.merge(recon[["subject_id", "sleep_date", "lifelog_date"] + rcols],
             on=["subject_id", "sleep_date", "lifelog_date"], how="left")


def bll(y, p):
    p = np.clip(p, EPS, 1 - EPS); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


P = np.column_stack([logit(mm[f"p_{t}"].values) for t in TARGETS])  # 7 OOF logits
R = mm[rcols].values.astype(float)
seeds = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173]

print("Cross-target + reconstruction stacking vs OOF baseline (interleaved CV, 10 seeds).")
print(f"  {'tgt':>4} | {'base ll':>8} | {'+xtarget':>9} | {'+xtgt+recon':>11} | {'best Δ':>8} | {'sign/N':>7} | {'level frac':>10}")
for ti, t in enumerate(TARGETS):
    y = mm[f"y_{t}"].values
    base, xt, xtr = [], [], []
    lvl_fracs = []
    sgn = 0; tot = 0
    for sd in seeds:
        for tr, va in H.interleaved_folds(mm, 5, seed=sd):
            ytr, yva = y[tr], y[va]
            base.append(bll(yva, mm[f"p_{t}"].values[va]))
            # +xtarget: logistic on all 7 OOF logits
            mu = P[tr].mean(0); sd_ = P[tr].std(0) + 1e-9
            Ptr = (P[tr] - mu) / sd_; Pva = (P[va] - mu) / sd_
            lr = LogisticRegression(C=1.0, max_iter=500).fit(Ptr, ytr)
            pxt = lr.predict_proba(Pva)[:, 1]; xt.append(bll(yva, pxt))
            # +xtarget+recon
            Rf = R.copy(); med = np.nanmedian(R[tr], 0); Rf = np.where(np.isnan(R), med, R)
            rmu = Rf[tr].mean(0); rsd = Rf[tr].std(0) + 1e-9
            Xtr = np.column_stack([Ptr, (Rf[tr] - rmu) / rsd]); Xva = np.column_stack([Pva, (Rf[va] - rmu) / rsd])
            lr2 = LogisticRegression(C=0.5, max_iter=500).fit(Xtr, ytr)
            px2 = lr2.predict_proba(Xva)[:, 1]; xtr.append(bll(yva, px2))
            # level/residual split of the +xtarget move vs base (per-subject)
            d = logit(pxt) - logit(mm[f"p_{t}"].values[va])
            subj = mm["subject_id"].values[va]
            lev = np.zeros_like(d)
            for s in np.unique(subj):
                lev[subj == s] = d[subj == s].mean()
            res = d - lev
            lf = (np.sum(lev ** 2) / (np.sum(d ** 2) + 1e-12))
            lvl_fracs.append(lf)
            sgn += int(np.mean(xt[-1]) < base[-1]); tot += 1
    b = np.mean(base); a1 = np.mean(xt); a2 = np.mean(xtr)
    best = min(a1, a2) - b
    sign = np.mean(np.array(xt) < np.array(base))
    flag = "  <<<" if (best < -0.003 and sign >= 0.7) else ""
    print(f"  {t:>4} | {b:>8.4f} | {a1:>9.4f} | {a2:>11.4f} | {best:>+8.4f} | {sign:>7.2f} | {np.mean(lvl_fracs):>10.2f}{flag}")

print("\n  Δ<0 & sign>=0.7 on a target => real cross-target/reconstruction per-row lever.")
print("  level-frac high => the gain is a transferable per-subject move; low => per-row lottery.")
print("  (mean Δ over 7 targets x how much it moves the mean toward 0.54.)")
