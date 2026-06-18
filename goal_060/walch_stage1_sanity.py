"""GOAL_060 Exp1 / Stage 1 — Walch sleep/wake SANITY on HR(BPM)+steps only (ETRI-transferable
channels), at a 60s epoch grid (matches ETRI minute granularity). NO raw accel (ETRI lacks it).

Question: with ONLY the channels ETRI can also provide (HR-bpm stats + step activity), at 60s
epochs, does a sleep/wake model trained on Walch's PSG labels reach a sane AUC (~0.8+)?
If yes -> the transferable feature set carries sleep signal, proceed to ETRI transfer.
If no  -> HR+steps@60s is too thin even with PSG labels (informs the ceiling question).

Features are label-free & subject-normalized so they are computable identically on ETRI:
  hr_mean/min/std within minute, hr_rel (subject median/IQR-normalized), step_sum, step_active.
Evaluated with LEAVE-ONE-SUBJECT-OUT (31 folds).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

DATA = Path(r"C:\Users\박광민\Documents\Codex\2026-06-18\handoff6-md-0-54-lb\data\walch_sleep_accel")
EPOCH = 60.0  # seconds (common grid with ETRI minute cadence)


def parse_pairs(path, sep=None):
    ts, val = [], []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split(sep) if sep else ln.split()
        if len(p) < 2:
            continue
        try:
            ts.append(float(p[0])); val.append(float(p[1]))
        except ValueError:
            continue
    return np.array(ts), np.array(val)


def subject_ids():
    return sorted({f.name.split("_")[0] for f in (DATA / "labels").glob("*_labeled_sleep.txt")})


def build_subject(sid):
    hr_ts, hr = parse_pairs(DATA / "heart_rate" / f"{sid}_heartrate.txt", sep=",")
    st_ts, st = parse_pairs(DATA / "steps" / f"{sid}_steps.txt", sep=",")
    lb_ts, lb = parse_pairs(DATA / "labels" / f"{sid}_labeled_sleep.txt", sep=None)  # space
    if len(lb) == 0 or len(hr) == 0:
        return None
    # label-free subject HR normalization (median / IQR over whole recording)
    hr_med = np.median(hr); hr_iqr = (np.percentile(hr, 75) - np.percentile(hr, 25)) or 1.0
    # 60s epoch grid over labeled region (labels are 30s @ t=0,30,60,...; majority within minute)
    t0 = 0.0
    tmax = lb_ts.max()
    edges = np.arange(t0, tmax + EPOCH, EPOCH)
    rows = []
    for e in edges[:-1]:
        lo, hi = e, e + EPOCH
        lm = (lb_ts >= lo) & (lb_ts < hi)
        if lm.sum() == 0:
            continue
        stages = lb[lm]
        sleep = float(np.mean(stages > 0) >= 0.5)  # majority sleep within the minute
        hm = (hr_ts >= lo) & (hr_ts < hi); sm = (st_ts >= lo) & (st_ts < hi)
        if hm.sum() == 0:
            continue
        h = hr[hm]
        feat = [
            h.mean(), h.min(), h.max(), h.std() if len(h) > 1 else 0.0,
            (h.mean() - hr_med) / hr_iqr,                       # subject-normalized HR level
            float(st[sm].sum()) if sm.sum() else 0.0,           # step activity in minute
            float((st[sm] > 0).any()) if sm.sum() else 0.0,     # any movement
        ]
        rows.append((feat, sleep))
    if not rows:
        return None
    X = np.array([r[0] for r in rows], float)
    y = np.array([r[1] for r in rows], float)
    return X, y


FEATS = ["hr_mean", "hr_min", "hr_max", "hr_std", "hr_rel", "step_sum", "step_active"]
subs = subject_ids()
data = {s: build_subject(s) for s in subs}
data = {s: d for s, d in data.items() if d is not None}
print(f"subjects usable: {len(data)}/{len(subs)}")
tot = sum(len(d[1]) for d in data.values()); pos = sum(d[1].sum() for d in data.values())
print(f"total 60s epochs: {tot}  sleep fraction: {pos/tot:.3f}")

# LOSO sleep/wake AUC
auc_lr, auc_gb = [], []
LGB = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=30,
           reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
keys = list(data.keys())
for held in keys:
    Xtr = np.vstack([data[s][0] for s in keys if s != held])
    ytr = np.concatenate([data[s][1] for s in keys if s != held])
    Xte, yte = data[held]
    if len(np.unique(yte)) < 2:
        continue
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    auc_lr.append(roc_auc_score(yte, lr.predict_proba(sc.transform(Xte))[:, 1]))
    gb = lgb.LGBMClassifier(**LGB).fit(Xtr, ytr)
    auc_gb.append(roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]))

print(f"\n=== Walch LOSO sleep/wake AUC (HR+steps @60s, ETRI-transferable feats) ===")
print(f"  LogisticReg : mean {np.mean(auc_lr):.3f}  median {np.median(auc_lr):.3f}  (n={len(auc_lr)})")
print(f"  LightGBM    : mean {np.mean(auc_gb):.3f}  median {np.median(auc_gb):.3f}")
# feature importance (gbm on all)
Xall = np.vstack([d[0] for d in data.values()]); yall = np.concatenate([d[1] for d in data.values()])
gb_all = lgb.LGBMClassifier(**LGB).fit(Xall, yall)
imp = sorted(zip(FEATS, gb_all.feature_importances_), key=lambda x: -x[1])
print("  feat importance:", {k: int(v) for k, v in imp})
sane = np.mean(auc_gb) >= 0.75 or np.mean(auc_lr) >= 0.75
print(f"\nSTAGE-1 VERDICT: {'SANE (>=0.75) -> proceed to ETRI transfer' if sane else 'WEAK (<0.75) -> HR+steps@60s thin'}")
