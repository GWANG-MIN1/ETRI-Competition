"""Strategy-doc Step 3/4 + Submission C: targetwise calibration / per-subject shrinkage.

Diagnostic (sp_00) showed base is OVER-CONFIDENT per-row on Q2/Q3/S2/S3 (oracle per-subject
constant beats base). So the transferable lever is to pull base toward the per-subject
level / lower sharpness. Test honestly (train-fold derived, applied to held-out):

  A. subj_shrink(kappa): logit(p) -> logit(p) + kappa*(logit(subj_train_mean) - logit(p))
  B. temp_scale: fit per-target temperature T on train fold; logit(p)/T  (global sharpness)
  C. subj_shrink toward recency-weighted subject mean (Q drift flavour)
  D. placebo: subject-shift assigned to a PERMUTED subject (kills true level, keeps magnitude)

Scored on BOTH test_faithful and interleaved CV, per target, multi-seed, vs base.
A lever is interesting only if delta<0 with high sign_frac AND beats placebo. NO LB fitting.
"""
import sys
from pathlib import Path
import numpy as np

KIT_DIR = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(HSJEPA_SRC)); sys.path.insert(0, str(KIT_DIR))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa

EPS = 1e-6
TARGETS = list(H.TARGETS)
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173]


def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))


def subj_train_mean(m, tr, t):
    g = m.iloc[tr].groupby("subject_id")[f"y_{t}"].mean()
    return g.to_dict()


def subj_recency_mean(m, tr, t, tau=21.0):
    trdf = m.iloc[tr]; rate = {}
    for s, idx in trdf.groupby("subject_id").groups.items():
        dd = trdf.loc[idx]
        last = dd["sleep_date"].max()
        w = np.exp(-(last - dd["sleep_date"]).dt.days.values / tau)
        rate[s] = float(np.sum(w * dd[f"y_{t}"].values) / np.sum(w))
    return rate


def est_subj_shrink(m, tr, va, t, kappa=0.3, recency=False, tau=21.0, placebo=False, seed=0):
    p = m[f"p_{t}"].values[va].copy()
    target_lvl = subj_recency_mean(m, tr, t, tau) if recency else subj_train_mean(m, tr, t)
    smean = subj_train_mean(m, tr, t)
    subj = m["subject_id"].values[va]
    uniq = list(np.unique(m["subject_id"].values))
    if placebo:
        rng = np.random.default_rng(seed * 977 + 3)
        perm = {s: ss for s, ss in zip(uniq, rng.permutation(uniq))}
    else:
        perm = {s: s for s in uniq}
    z = logit(p)
    out = z.copy()
    for i, s in enumerate(subj):
        src = perm[s]
        lvl = logit(np.clip(target_lvl.get(src, smean.get(src, p.mean())), EPS, 1 - EPS))
        out[i] = z[i] + kappa * (lvl - z[i])
    return np.clip(sig(out), EPS, 1 - EPS)


def fit_temp(m, tr, t):
    """1-param temperature minimizing train-fold logloss: logit(p)/T."""
    y = m[f"y_{t}"].values[tr]; z = logit(m[f"p_{t}"].values[tr])
    best, bestT = 1e9, 1.0
    for T in np.linspace(0.6, 2.2, 33):
        ll = H.bll(y, sig(z / T))
        if ll < best:
            best, bestT = ll, T
    return bestT


def est_temp(m, tr, va, t, **kw):
    T = fit_temp(m, tr, t)
    return np.clip(sig(logit(m[f"p_{t}"].values[va]) / T), EPS, 1 - EPS)


def eval_estimator(m, est, proxy, **kw):
    res = {t: {"d": [], "imp": 0} for t in TARGETS}
    for sd in SEEDS:
        if proxy == "test_faithful":
            held = H.test_faithful_mask(m, sd)
            splits = [(np.flatnonzero(~held), np.flatnonzero(held))]
        else:
            splits = H.interleaved_folds(m, 5, sd)
        for t in TARGETS:
            bl, sl = [], []
            for tr, va in splits:
                y = m[f"y_{t}"].values[va]
                bl.append(H.bll(y, m[f"p_{t}"].values[va]))
                sl.append(H.bll(y, est(m, tr, va, t, seed=sd, **kw)))
            b, s_ = np.mean(bl), np.mean(sl)
            res[t]["d"].append(s_ - b); res[t]["imp"] += int(s_ < b)
    out = {}
    for t in TARGETS:
        d = np.array(res[t]["d"])
        out[t] = (float(d.mean()), res[t]["imp"] / len(SEEDS))
    return out


def show(title, out):
    print(f"\n--- {title} ---")
    print("  tgt |  delta   | sign  (delta<0=better)")
    for t in TARGETS:
        d, sf = out[t]
        flag = "  <<<" if d < -0.0008 and sf >= 0.7 else ("  <" if d < 0 and sf >= 0.6 else "")
        print(f"  {t}  | {d:+.5f} | {sf:.2f}{flag}")


if __name__ == "__main__":
    m = K.base()
    print("base mean test_faithful:", round(K.meanll(K.score_perrow(m, np.column_stack([m[f'p_{t}'].values for t in TARGETS]))), 5))

    for proxy in ("test_faithful", "interleaved"):
        print(f"\n================ PROXY = {proxy} ================")
        # temperature scaling
        show(f"B. temp_scale [{proxy}]", eval_estimator(m, est_temp, proxy))
        # subject shrink sweep
        for kap in (0.1, 0.2, 0.3, 0.4, 0.5):
            show(f"A. subj_shrink k={kap} [{proxy}]", eval_estimator(m, est_subj_shrink, proxy, kappa=kap))
        # recency-mean shrink (Q flavour)
        for kap in (0.2, 0.4):
            show(f"C. subj_recency_shrink k={kap} [{proxy}]", eval_estimator(m, est_subj_shrink, proxy, kappa=kap, recency=True))
        # placebo control at the most promising kappa
        show(f"D. PLACEBO subj_shrink k=0.3 [{proxy}]", eval_estimator(m, est_subj_shrink, proxy, kappa=0.3, placebo=True))
    print("\nDONE")
