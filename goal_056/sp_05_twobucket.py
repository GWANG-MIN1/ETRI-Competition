"""Test the divergence-workflow's standout idea: TWO-BUCKET (interleaved vs future)
date-step overshoot on Q2/Q3.

Premise: the deployed best applies ONE sigma (=0.8) to the per-subject Q2/Q3 recency
logit-step uniformly. But the test splits into interleaved-dated rows (between train days,
low drift -> want small sigma) and future-dated rows (after all train days, max drift ->
want large sigma). The bucket label = (row date > subject's train-max date) is OBSERVED
(zero estimation noise), so a 2-value-per-subject step stays in the transferable level class.

Falsification (forward-CV on the 450 train rows):
  Split each test_faithful held-out into FUTURE block (latest 15%) vs INTERLEAVED block.
  Sweep sigma; find the sigma that minimizes logloss on EACH block separately.
  PASS premise iff future-optimal sigma > interleaved-optimal sigma (robustly, multi-seed)
  AND a PLACEBO (random bucket relabel) shows NO such separation.
NOTE: only Q2/Q3 (drift above the 0.12 transfer threshold). NO LB fitting.
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
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173, 197, 211]
SIGMAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))


def split_future_inter(m, seed, future_frac=0.15, inter_frac=0.25):
    """Replicate test_faithful_mask but return (future_idx, inter_idx) separately."""
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


def run(t, placebo=False):
    m = K.base()
    # accumulate logloss(sigma) for each block across seeds
    fut_ll = {sg: [] for sg in SIGMAS}
    int_ll = {sg: [] for sg in SIGMAS}
    for seed in SEEDS:
        fidx, iidx = split_future_inter(m, seed)
        held = np.concatenate([fidx, iidx])
        tr = np.setdiff1d(np.arange(len(m)), held)
        step = recency_step(m, tr, t)
        subj = m["subject_id"].values
        y = m[f"y_{t}"].values
        z0 = logit(m[f"p_{t}"].values)
        st = np.array([step.get(s, 0.0) for s in subj])
        if placebo:
            # random bucket relabel: shuffle which held rows are 'future' vs 'interleaved'
            rng = np.random.default_rng(seed * 31 + 1)
            allheld = held.copy(); rng.shuffle(allheld)
            fidx = allheld[:len(fidx)]; iidx = allheld[len(fidx):]
        for sg in SIGMAS:
            pz = z0 + sg * st
            fut_ll[sg].append(H.bll(y[fidx], sig(pz[fidx])))
            int_ll[sg].append(H.bll(y[iidx], sig(pz[iidx])))
    fut_mean = {sg: np.mean(fut_ll[sg]) for sg in SIGMAS}
    int_mean = {sg: np.mean(int_ll[sg]) for sg in SIGMAS}
    fbest = min(SIGMAS, key=lambda s: fut_mean[s])
    ibest = min(SIGMAS, key=lambda s: int_mean[s])
    tag = "PLACEBO" if placebo else "REAL"
    print(f"\n--- {t} [{tag}] forward-CV logloss by sigma ---")
    print("  sigma : " + " ".join(f"{s:>5.1f}" for s in SIGMAS))
    print("  FUTURE: " + " ".join(f"{fut_mean[s]:.4f}" for s in SIGMAS) + f"   -> opt sigma={fbest}")
    print("  INTER : " + " ".join(f"{int_mean[s]:.4f}" for s in SIGMAS) + f"   -> opt sigma={ibest}")
    # gain of optimal-per-block vs flat 0.8 (weighted 37.6/62.4 like the real test)
    flat = 0.376 * fut_mean[0.8] + 0.624 * int_mean[0.8]
    step2 = 0.376 * fut_mean[fbest] + 0.624 * int_mean[ibest]
    print(f"  blended(37.6/62.4): flat0.8={flat:.5f}  2-bucket(opt)={step2:.5f}  gain={step2-flat:+.5f}")
    return dict(t=t, fbest=fbest, ibest=ibest, sep=fbest - ibest, gain=step2 - flat,
                fut=fut_mean, int=int_mean)


if __name__ == "__main__":
    print("Two-bucket date-step overshoot test (forward-CV premise check)")
    out = {}
    for t in ("Q2", "Q3"):
        out[(t, "real")] = run(t, placebo=False)
        out[(t, "plac")] = run(t, placebo=True)
    print("\n=== SUMMARY (premise: future opt-sigma > interleaved opt-sigma, real but NOT placebo) ===")
    for t in ("Q2", "Q3"):
        r = out[(t, "real")]; p = out[(t, "plac")]
        verdict = "PASS" if (r["sep"] > 0 and r["gain"] < -0.0003 and p["sep"] <= r["sep"]) else "FAIL"
        print(f"  {t}: real sep={r['sep']:+.1f} gain={r['gain']:+.5f} | placebo sep={p['sep']:+.1f} gain={p['gain']:+.5f}  => {verdict}")
    print("\nDONE")
