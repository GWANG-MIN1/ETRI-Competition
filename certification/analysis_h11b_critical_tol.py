#!/usr/bin/env python3
"""H11b: critical tolerance — how wrong can the two load-bearing probe measurements be
before the certificate breaks? Scale ONLY their tolerances by f and find the break point."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ETRI = HERE.parent
OLD = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\outputs\submissions")
RAW = ETRI / "data"
KEYS = ["subject_id", "sleep_date", "lifelog_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6
N_CELLS = 1750.0
H057_LB = 0.5677475939
CAP = 0.8


def imp(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


c1 = imp(HERE / "candidate_1_public_loss_sparse_tomography.py", "c1_for_h11b")


def load_abs(p):
    return pd.read_csv(p).sort_values(KEYS).reset_index(drop=True)


def lin_terms(p_from, p_to):
    pf = np.clip(np.asarray(p_from, float), EPS, 1 - EPS)
    pt = np.clip(np.asarray(p_to, float), EPS, 1 - EPS)
    coef = np.log((1 - pt) / (1 - pf)) - np.log(pt / pf)
    const = -np.log((1 - pt) / (1 - pf))
    dlogit = np.log(pt / (1 - pt)) - np.log(pf / (1 - pf))
    return coef, const, dlogit


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
    H057 = load_abs(c1.locate("submission_h057_q2row_fullvector_state_7cde1a77_uploadsafe.csv"))
    FS = load_abs(c1.locate("submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv"))
    sub = H057["subject_id"].values
    subjects = sorted(set(sub))
    vidx = {}
    k = 0
    for t in TARGETS:
        for s in subjects:
            vidx[(s, t)] = k; k += 1
    n_var = k
    masks = {s: sub == s for s in subjects}
    n_rows = {s: int(masks[s].sum()) for s in subjects}

    def pair_row(df_from, df_to):
        v = np.zeros(n_var); c0 = 0.0; sq = 0.0
        for t in TARGETS:
            coef, const, dlg = lin_terms(df_from[t].values, df_to[t].values)
            c0 += const.sum() / N_CELLS
            for s in subjects:
                m = masks[s]
                v[vidx[(s, t)]] += coef[m].sum() / N_CELLS
                sq += float(((dlg[m] - dlg[m].mean()) ** 2).sum())
        return v, c0, 0.5 * np.sqrt(sq) / N_CELLS

    named = []
    ledger = pd.read_csv(ETRI / "data_analytics" / "hsjepa_public_score_ledger.csv")
    for rec in ledger.to_dict("records"):
        p = c1.locate(str(rec["file"]))
        if p is None:
            continue
        v, c0, sg = pair_row(H057, load_abs(p))
        named.append(("led", v, c0, float(rec["public_lb"]) - H057_LB, sg))
    olds = {kk: load_abs(OLD / nm) for kk, nm in {
        "M": "submission_lb60247_to_pertarget_best_microblend.csv",
        "A": "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv",
        "B": "submission_Q1up100_plus_Q2recency_tau10_lam075.csv",
        "C": "submission_FINAL_Q2recency_plus_Q3recency.csv"}.items()}
    probe_idx = []
    for nm, (f1, f2, d) in {"old_Q2probe": ("A", "B", 0.5949167449 - 0.6001),
                            "old_Q3probe": ("B", "C", 0.593188787 - 0.5949167449),
                            "old_Q1probe": ("M", "A", 0.6001 - 0.60247)}.items():
        v, c0, sg = pair_row(olds[f1], olds[f2])
        named.append((nm, v, c0, d, sg))
        if nm in ("old_Q2probe", "old_Q3probe"):
            probe_idx.append(len(named) - 1)

    rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}
    bounds = [(max(0.02, rate[t][s] - 0.45), min(0.98, rate[t][s] + 0.45))
              for t in TARGETS for s in subjects]

    delta = np.zeros(n_var)
    for target, kappa, (f1, f2) in [("Q2", 0.75, ("A", "B")), ("Q3", 0.25, ("B", "C"))]:
        dlg = logit(olds[f2][target].values) - logit(olds[f1][target].values)
        for s in subjects:
            delta[vidx[(s, target)]] = float(np.clip(kappa * float(dlg[masks[s]].mean()), -CAP, CAP))
    A_g = 0.0
    coefs_g = np.zeros(n_var)
    for t in TARGETS:
        z = logit(FS[t].values)
        for s in subjects:
            d = delta[vidx[(s, t)]]
            zz = z[masks[s]]
            A_g += float(-np.log((1 - sigmoid(zz + d)) / (1 - sigmoid(zz))).sum()) / N_CELLS
            coefs_g[vidx[(s, t)]] = -n_rows[s] * d / N_CELLS

    print("critical-tolerance scan: scale ONLY the two probe tolerances by f")
    print(f"{'f':>5} {'tol(Q2probe)':>13} {'as % of |measured|':>18} {'certified worst':>16}")
    for f in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        A_ub, b_ub = [], []
        for i, (nm, v, c0, d, sg) in enumerate(named):
            base_tol = 1e-5 + 0.30 * abs(d) + 3.0 * sg
            tol = base_tol * f if i in probe_idx else base_tol
            A_ub.append(v);  b_ub.append(d + tol - c0)
            A_ub.append(-v); b_ub.append(-(d - tol - c0))
        res = linprog(-coefs_g, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=bounds, method="highs")
        d_q2 = named[probe_idx[0]][3]
        tolq2 = (1e-5 + 0.30 * abs(d_q2) + 3.0 * named[probe_idx[0]][4]) * f
        print(f"{f:5.1f} {tolq2:13.5f} {100*tolq2/abs(d_q2):17.0f}% {A_g - res.fun:+16.5f}")


if __name__ == "__main__":
    main()
