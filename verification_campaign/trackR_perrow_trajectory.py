#!/usr/bin/env python3
"""TRACK R — per-row signal re-attempt, drift-aligned (the one axis that transfers).

The overshoot is a UNIFORM per-subject shift. But drift is TEMPORAL: recent/future rows have
drifted more. A per-row LINEAR-TRAJECTORY shift (each row moved toward its time-extrapolated
rate) is per-row but DRIFT-ALIGNED, so it could transfer where within-subject reranks don't.
Test on test-faithful CV (holds out FUTURE rows = the most-drifted), split future vs interleaved.
Compare: baseline | uniform recency shift | per-row linear trajectory shift.
If the trajectory beats uniform on FUTURE held-out Q2/Q3, that is a NEW transferable per-row gain.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

CORE = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(CORE))
import hsjepa_core as H  # noqa: E402

EPS = 1e-6
m = H.load()


def norm_time(df):
    x = np.zeros(len(df))
    for s, idx in df.groupby("subject_id").groups.items():
        d = df.loc[idx]
        days = (d["sleep_date"] - d["sleep_date"].min()).dt.days.values.astype(float)
        rng = max(days.max(), 1)
        x[[df.index.get_loc(i) for i in idx]] = days / rng
    return x


M = m.reset_index(drop=True)
XT = norm_time(M)


def est_uniform_recency(m, tr, va, t, lam=1.0, tau=21.0, **kw):
    p = m[f"p_{t}"].values[va].copy()
    trdf = m.iloc[tr]
    rate, smean = {}, {}
    for s, idx in trdf.groupby("subject_id").groups.items():
        d = trdf.loc[idx]
        last = d["sleep_date"].max()
        w = np.exp(-(last - d["sleep_date"]).dt.days.values / tau)
        rate[s] = float(np.sum(w * d[f"y_{t}"].values) / np.sum(w))
        smean[s] = float(d[f"y_{t}"].mean())
    subj = m["subject_id"].values[va]
    shift = np.array([lam * (rate.get(s, 0.5) - smean.get(s, 0.5)) for s in subj])
    return np.clip(p + shift, EPS, 1 - EPS)


def est_trajectory(m, tr, va, t, lam=1.0, **kw):
    """per-row: fit per-subject rate ~ a+b*x on train fold; shift each held-out row toward
    its extrapolated a+b*x (vs the subject mean)."""
    p = m[f"p_{t}"].values[va].copy()
    trdf = m.iloc[tr]
    coef, smean = {}, {}
    for s, idx in trdf.groupby("subject_id").groups.items():
        d = trdf.loc[idx]
        x = XT[idx]
        y = d[f"y_{t}"].values.astype(float)
        smean[s] = float(y.mean())
        if x.std() > 1e-9 and len(x) >= 6:
            b, a = np.polyfit(x, y, 1)
            coef[s] = (a, b)
        else:
            coef[s] = (smean[s], 0.0)
    subj = m["subject_id"].values[va]
    xva = XT[va]
    shift = np.zeros(len(va))
    for i, (s, xi) in enumerate(zip(subj, xva)):
        a, b = coef.get(s, (smean.get(s, 0.5), 0.0))
        target = np.clip(a + b * xi, 0.02, 0.98)
        shift[i] = lam * (target - smean.get(s, 0.5))
    return np.clip(p + shift, EPS, 1 - EPS)


# test-faithful CV, separate FUTURE vs INTERLEAVED held-out scoring
seeds = [11, 23, 37, 51, 67, 83, 101, 131]
TARGETS = H.TARGETS
res = {t: {"uni_f": [], "traj_f": [], "uni_i": [], "traj_i": [], "base_f": [], "base_i": []} for t in TARGETS}
for sd in seeds:
    held = H.test_faithful_mask(M, sd)
    # split held into future (latest 15%) vs interleaved
    fut = np.zeros(len(M), bool)
    for s, idx in M.groupby("subject_id").groups.items():
        idx = np.array(sorted(idx, key=lambda i: M.at[i, "sleep_date"]))
        nf = max(1, int(round(len(idx) * 0.15)))
        fut[idx[-nf:]] = True
    tr = np.flatnonzero(~held)
    va = np.flatnonzero(held)
    va_f = va[fut[va]]
    va_i = va[~fut[va]]
    for t in TARGETS:
        for lab, vsub in [("f", va_f), ("i", va_i)]:
            if len(vsub) == 0:
                continue
            y = M[f"y_{t}"].values[vsub]
            res[t][f"base_{lab}"].append(H.bll(y, M[f"p_{t}"].values[vsub]))
            res[t][f"uni_{lab}"].append(H.bll(y, est_uniform_recency(M, tr, vsub, t)))
            res[t][f"traj_{lab}"].append(H.bll(y, est_trajectory(M, tr, vsub, t)))

print("Test-faithful CV: delta vs baseline (neg=better), split FUTURE vs INTERLEAVED held-out.")
print("Does per-row TRAJECTORY beat UNIFORM recency on the drifted FUTURE rows?\n")
print(f"  {'tgt':>4} | {'FUTURE uni':>11} {'FUTURE traj':>12} {'traj-uni':>9} | {'INTER uni':>10} {'INTER traj':>11}")
for t in TARGETS:
    bf = np.mean(res[t]["base_f"]); bi = np.mean(res[t]["base_i"])
    uf = np.mean(res[t]["uni_f"]) - bf; tf = np.mean(res[t]["traj_f"]) - bf
    ui = np.mean(res[t]["uni_i"]) - bi; ti = np.mean(res[t]["traj_i"]) - bi
    mark = " <<<" if (tf < uf - 0.0005) else ""
    print(f"  {t:>4} | {uf:>+11.4f} {tf:>+12.4f} {tf-uf:>+9.4f} | {ui:>+10.4f} {ti:>+11.4f}{mark}")
print("\n  traj-uni < 0 on FUTURE Q2/Q3 => per-row trajectory adds transferable drift signal.")
print("  >= 0 => uniform per-subject shift already captures all transferable drift (per-row dead).")
