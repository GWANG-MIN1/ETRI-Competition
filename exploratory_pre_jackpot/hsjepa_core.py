"""HS-JEPA shared core: data + test-faithful CV + module level-estimators.

Philosophy (hard-won from memory): only LEVEL/PRIOR shifts transfer to the 250-row
interleaved LB; per-row reranks are sign-unreliable. So every team module is turned
into a *level estimator* (per-subject or global base-rate shift) and measured honestly.

CV proxies:
  * test_faithful : per subject, hold out latest ~15% (future) + random ~25% (interleaved)
                    ~= the real test composition (37.6% future / 62.4% interleaved).
  * interleaved   : random within-subject K-fold (the proxy that correctly called the
                    S234 rerank LB failure).
Baseline per-row preds = cached unified OOF (leakage-free). Shifts are computed from
TRAIN-FOLD labels only (causal), applied to HELD-OUT preds, scored vs true labels.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "raw"
KEYS = ["subject_id", "sleep_date", "lifelog_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6


def bll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def load():
    """Return merged frame: KEYS + true labels (y_*) + baseline OOF preds (p_*)."""
    lbl = pd.read_csv(RAW / "ch2026_metrics_train.csv")
    oof = pd.read_csv(ROOT / "outputs/models/unified/subject_blocked_time/cat/oof.csv")
    for d in (lbl, oof):
        d["sleep_date"] = pd.to_datetime(d["sleep_date"])
        d["lifelog_date"] = pd.to_datetime(d["lifelog_date"])
    if "__oof_valid" in oof.columns:
        oof = oof[oof["__oof_valid"] == 1]
    oof = oof.rename(columns={t: f"p_{t}" for t in TARGETS})
    lbl = lbl.rename(columns={t: f"y_{t}" for t in TARGETS})
    m = lbl.merge(oof[KEYS + [f"p_{t}" for t in TARGETS]], on=KEYS, how="inner")
    m = m.sort_values(["subject_id", "sleep_date"]).reset_index(drop=True)
    return m


def test_faithful_mask(m, seed, future_frac=0.15, inter_frac=0.25):
    """Held-out = latest `future_frac` per subject (future) + random `inter_frac` of the
    rest (interleaved). Mimics the 38/62 future/interleaved test split."""
    rng = np.random.default_rng(seed)
    held = np.zeros(len(m), bool)
    for s, idx in m.groupby("subject_id").groups.items():
        idx = np.array(sorted(idx, key=lambda i: m.at[i, "sleep_date"]))
        n = len(idx)
        n_fut = max(1, int(round(n * future_frac)))
        fut = idx[-n_fut:]
        rest = idx[:-n_fut]
        n_int = max(1, int(round(n * inter_frac)))
        inter = rng.choice(rest, size=min(n_int, len(rest)), replace=False)
        held[fut] = True
        held[inter] = True
    return held


def interleaved_folds(m, n_splits, seed):
    rng = np.random.default_rng(seed)
    fold = np.full(len(m), -1)
    for s in m["subject_id"].unique():
        idx = np.flatnonzero(m["subject_id"].values == s)
        perm = rng.permutation(idx)
        for k, chunk in enumerate(np.array_split(perm, n_splits)):
            fold[chunk] = k
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_splits)]


# ---------------------------------------------------------------------------
# Module level-estimators. Each returns shifted held-out preds for one target.
# Signature: fn(m, tr_idx, va_idx, t, **kw) -> shifted preds on va (np array len=len(va))
# ---------------------------------------------------------------------------

def est_baseline(m, tr, va, t, **kw):
    return m[f"p_{t}"].values[va].copy()


def _subj_train_mean(m, tr, t):
    g = m.iloc[tr].groupby("subject_id")[f"y_{t}"].mean()
    return g.to_dict()


def est_subject_recency(m, tr, va, t, tau=21.0, lam=1.0, **kw):
    """Q2/Q3 proven lever: shift toward subject's recency-weighted recent train label rate."""
    p = m[f"p_{t}"].values[va].copy()
    trdf = m.iloc[tr]
    rate = {}
    for s, idx in trdf.groupby("subject_id").groups.items():
        d = trdf.loc[idx]
        last = d["sleep_date"].max()
        w = np.exp(-(last - d["sleep_date"]).dt.days.values / tau)
        rate[s] = float(np.sum(w * d[f"y_{t}"].values) / np.sum(w))
    smean = _subj_train_mean(m, tr, t)
    subj = m["subject_id"].values[va]
    shift = np.array([lam * (rate.get(s, smean.get(s, p.mean())) - smean.get(s, p.mean())) for s in subj])
    return np.clip(p + shift, EPS, 1 - EPS)


def est_cohort_shrink(m, tr, va, t, shrink=0.3, **kw):
    """V131C regression-to-cohort-mean: pull each subject's level toward the cohort mean
    by `shrink` (mean reversion of extremes). Shift = shrink*(cohort - subj_mean)."""
    p = m[f"p_{t}"].values[va].copy()
    smean = _subj_train_mean(m, tr, t)
    cohort = float(np.mean(list(smean.values())))
    subj = m["subject_id"].values[va]
    shift = np.array([shrink * (cohort - smean.get(s, cohort)) for s in subj])
    return np.clip(p + shift, EPS, 1 - EPS)


def est_transition_drift(m, tr, va, t, lam=0.5, **kw):
    """State-Transition Verb / Lag-Hysteresis: per-subject late-minus-early train slope =
    'getting better / falling apart' direction; nudge held-out level along it."""
    p = m[f"p_{t}"].values[va].copy()
    trdf = m.iloc[tr]
    drift = {}
    for s, idx in trdf.groupby("subject_id").groups.items():
        d = trdf.loc[idx].sort_values("sleep_date")
        if len(d) < 6:
            drift[s] = 0.0
            continue
        half = len(d) // 2
        early = d[f"y_{t}"].values[:half].mean()
        late = d[f"y_{t}"].values[half:].mean()
        drift[s] = late - early
    subj = m["subject_id"].values[va]
    shift = np.array([lam * drift.get(s, 0.0) for s in subj])
    return np.clip(p + shift, EPS, 1 - EPS)


def est_global_shift(m, tr, va, t, delta=0.0, **kw):
    """Global level shift (Q-up / S-down style). NOTE: public-prior elevation is NOT in
    train labels, so CV will typically PENALIZE this -> demonstrates why Q-up is a
    LB-PROBE bet, not a CV-visible gain."""
    p = m[f"p_{t}"].values[va].copy()
    return np.clip(p + delta, EPS, 1 - EPS)


def est_rerank_control(m, tr, va, t, seed=0, **kw):
    """Negative control: mean-preserving within-subject shuffle of preds (pure per-row)."""
    p = m[f"p_{t}"].values[va].copy()
    rng = np.random.default_rng(seed)
    subj = m["subject_id"].values[va]
    out = p.copy()
    for s in np.unique(subj):
        mask = subj == s
        vals = p[mask]
        out[mask] = rng.permutation(vals)
    return out


# ---------------------------------------------------------------------------
def cv_eval(estimator, proxy="test_faithful", n_seeds=12, **kw):
    """Return per-target {delta_mean, sign_frac, base_ll} vs baseline, averaged over seeds."""
    m = kw.pop("frame", None)
    if m is None:
        m = load()
    seeds = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173, 197, 211][:n_seeds]
    res = {t: {"d": [], "imp": 0, "base": []} for t in TARGETS}
    for sd in seeds:
        if proxy == "test_faithful":
            held = test_faithful_mask(m, sd)
            splits = [(np.flatnonzero(~held), np.flatnonzero(held))]
        else:
            splits = interleaved_folds(m, n_splits=5, seed=sd)
        for t in TARGETS:
            base_lls, shift_lls = [], []
            for tr, va in splits:
                y = m[f"y_{t}"].values[va]
                base_lls.append(bll(y, m[f"p_{t}"].values[va]))
                shift_lls.append(bll(y, estimator(m, tr, va, t, seed=sd, **kw)))
            b = np.mean(base_lls); s_ = np.mean(shift_lls)
            res[t]["d"].append(s_ - b)
            res[t]["base"].append(b)
            res[t]["imp"] += int(s_ < b)
    out = {}
    n = len(seeds)
    for t in TARGETS:
        d = np.array(res[t]["d"])
        out[t] = {"delta": float(d.mean()), "sd": float(d.std()),
                  "sign_frac": res[t]["imp"] / n, "base_ll": float(np.mean(res[t]["base"]))}
    return out


def fmt(out, label):
    print(f"\n--- {label} ---")
    print("  tgt | base_ll |  delta  | sd     | imp/N  (delta<0 = improvement)")
    for t in TARGETS:
        o = out[t]
        flag = "  <<<" if (o["delta"] < -0.0008 and o["sign_frac"] >= 0.75) else ""
        print(f"  {t}  | {o['base_ll']:.4f}  | {o['delta']:+.4f} | {o['sd']:.4f} | {o['sign_frac']:.2f}{flag}")
