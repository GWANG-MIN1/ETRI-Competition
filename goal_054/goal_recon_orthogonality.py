#!/usr/bin/env python3
"""GOAL — does the phone sleep-reconstruction ADD orthogonal signal to the team model on S?

Even if reconstruction loses standalone, it could help if orthogonal. Test per S target via
interleaved CV: logloss of (a) baseline = team OOF logit, vs (b) [OOF logit + recon features]
in a small logistic model. If (b) << (a) on S1/S2, reconstruction is a real additive lever
worth a full build. If it ties, reconstruction is subsumed by the 5245-col store -> dead end.
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
recon = pd.read_parquet(HERE / "sleep_recon_train.parquet")  # from v1 (has tst/se/sol/waso/tib)
# rebuild from v2 if needed; v1 saved tr2 with recon cols
m = H.load()
feat_cols = [c for c in ["tst", "tib", "se", "sol", "waso", "onset_h"] if c in recon.columns]
recon["sleep_date"] = pd.to_datetime(recon["sleep_date"]); recon["lifelog_date"] = pd.to_datetime(recon["lifelog_date"])
mm = m.merge(recon[["subject_id", "sleep_date", "lifelog_date"] + feat_cols],
             on=["subject_id", "sleep_date", "lifelog_date"], how="left")


def bll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


print(f"recon features used: {feat_cols}")
print("\n=== CV logloss: OOF baseline vs OOF+recon (interleaved, 8 seeds) ===")
print(f"  {'tgt':>4} | {'base ll':>8} | {'+recon ll':>9} | {'delta':>8} | {'sign/N':>7}")
seeds = [11, 23, 37, 51, 67, 83, 101, 131]
for t in ["S1", "S2", "S3", "S4"]:
    bds, ads, imp = [], [], 0
    for sd in seeds:
        for tr, va in H.interleaved_folds(mm, 5, seed=sd):
            ytr = mm[f"y_{t}"].values[tr]; yva = mm[f"y_{t}"].values[va]
            base_tr = logit(mm[f"p_{t}"].values[tr]); base_va = logit(mm[f"p_{t}"].values[va])
            # baseline: just use OOF pred
            bds.append(bll(yva, mm[f"p_{t}"].values[va]))
            # + recon: logistic on [oof logit, recon feats] (impute median, standardize-ish)
            X = mm[feat_cols].values.astype(float)
            med = np.nanmedian(X[tr], axis=0)
            Xtr = np.where(np.isnan(X[tr]), med, X[tr]); Xva = np.where(np.isnan(X[va]), med, X[va])
            mu = Xtr.mean(0); sd_ = Xtr.std(0) + 1e-9
            Xtr = (Xtr - mu) / sd_; Xva = (Xva - mu) / sd_
            Xtr2 = np.column_stack([base_tr, Xtr]); Xva2 = np.column_stack([base_va, Xva])
            lr = LogisticRegression(C=0.5, max_iter=500).fit(Xtr2, ytr)
            p = lr.predict_proba(Xva2)[:, 1]
            ads.append(bll(yva, p))
    b = np.mean(bds); a = np.mean(ads); imp = np.mean(np.array(ads) < np.array(bds))
    flag = "  <<< ADDS" if (a < b - 0.003 and imp >= 0.7) else ""
    print(f"  {t:>4} | {b:>8.4f} | {a:>9.4f} | {a-b:>+8.4f} | {imp:>7.2f}{flag}")

print("\n  delta<0 (& sign>=0.7) on S1/S2 => reconstruction adds orthogonal transferable signal.")
print("  delta~0 => subsumed by the team feature store; the easy-reconstruction path is closed.")
