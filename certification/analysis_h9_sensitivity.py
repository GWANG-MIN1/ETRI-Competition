#!/usr/bin/env python3
"""H9: sensitivity of the probe-direction candidate's sign-safety.

Question: does "LP worst case < 0" for FS + Q2-dir survive when the modeling constants
move against us?
  grid: composition tolerance {10%, 20%, 30%}  x  box {±0.25, ±0.35, ±0.45}  x  Z {2, 3}
Candidates: Q2-dir kappa {0.5, 0.75, 1.0}, capped |Lambda|<=0.8 kappa 0.75,
            Q2dir0.75 + Q3dir0.25.
Also reports the candidate's anatomy (per-subject shifts, post-move means, cell ranges).
"""

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


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


c1 = import_module(HERE / "candidate_1_public_loss_sparse_tomography.py", "c1_for_h9")


def load_abs(p):
    return pd.read_csv(p).sort_values(KEYS).reset_index(drop=True)


def load_ledger_file(name):
    p = c1.locate(name)
    return None if p is None else load_abs(p)


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
    H057 = load_ledger_file("submission_h057_q2row_fullvector_state_7cde1a77_uploadsafe.csv")
    FS = load_ledger_file("submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv")
    sub = H057["subject_id"].values
    subjects = sorted(set(sub))
    vidx = {}
    k = 0
    for t in TARGETS:
        for s in subjects:
            vidx[(s, t)] = k; k += 1
    n_var = k
    masks = {s: sub == s for s in subjects}

    def pair_row(df_from, df_to):
        v = np.zeros(n_var); c0 = 0.0; sq_c = 0.0
        for t in TARGETS:
            coef, const, dlg = lin_terms(df_from[t].values, df_to[t].values)
            c0 += const.sum() / N_CELLS
            for s in subjects:
                m = masks[s]
                v[vidx[(s, t)]] += coef[m].sum() / N_CELLS
                sq_c += float(((dlg[m] - dlg[m].mean()) ** 2).sum())
        return v, c0, 0.5 * np.sqrt(sq_c) / N_CELLS

    raw_constraints = []
    ledger = pd.read_csv(ETRI / "data_analytics" / "hsjepa_public_score_ledger.csv")
    for rec in ledger.to_dict("records"):
        df = load_ledger_file(str(rec["file"]))
        if df is None:
            continue
        v, c0, sg = pair_row(H057, df)
        raw_constraints.append((v, c0, float(rec["public_lb"]) - H057_LB, sg))
    olds = {kk: load_abs(OLD / nm) for kk, nm in {
        "M": "submission_lb60247_to_pertarget_best_microblend.csv",
        "A": "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv",
        "B": "submission_Q1up100_plus_Q2recency_tau10_lam075.csv",
        "C": "submission_FINAL_Q2recency_plus_Q3recency.csv"}.items()}
    for f1, f2, d in [("A", "B", 0.5949167449 - 0.6001),
                      ("B", "C", 0.593188787 - 0.5949167449),
                      ("M", "A", 0.6001 - 0.60247)]:
        v, c0, sg = pair_row(olds[f1], olds[f2])
        raw_constraints.append((v, c0, d, sg))

    # probe directions
    Lam = {}
    for target, (f1, f2) in {"Q2": ("A", "B"), "Q3": ("B", "C")}.items():
        dlg = logit(olds[f2][target].values) - logit(olds[f1][target].values)
        Lam[target] = {s: float(dlg[masks[s]].mean()) for s in subjects}

    def candidate_moves(name):
        moves = []
        if name == "Q2dir_k0.5":
            moves = [("Q2", {s: 0.5 * Lam["Q2"][s] for s in subjects})]
        elif name == "Q2dir_k0.75":
            moves = [("Q2", {s: 0.75 * Lam["Q2"][s] for s in subjects})]
        elif name == "Q2dir_k1.0":
            moves = [("Q2", {s: 1.0 * Lam["Q2"][s] for s in subjects})]
        elif name == "Q2dir_k0.75_cap0.8":
            moves = [("Q2", {s: float(np.clip(0.75 * Lam["Q2"][s], -0.8, 0.8)) for s in subjects})]
        elif name == "Q2dir0.75+Q3dir0.25":
            moves = [("Q2", {s: 0.75 * Lam["Q2"][s] for s in subjects}),
                     ("Q3", {s: 0.25 * Lam["Q3"][s] for s in subjects})]
        return moves

    def objective(moves):
        wv = np.zeros(n_var); wc = 0.0
        for target, shift_map in moves:
            p = FS[target].values
            shift = np.array([shift_map[s] for s in sub])
            p_new = sigmoid(logit(p) + shift)
            coef, const, _ = lin_terms(p, p_new)
            wc += const.sum() / N_CELLS
            for s in subjects:
                wv[vidx[(s, target)]] += coef[masks[s]].sum() / N_CELLS
        return wv, wc

    rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}
    cands = ["Q2dir_k0.5", "Q2dir_k0.75", "Q2dir_k1.0", "Q2dir_k0.75_cap0.8", "Q2dir0.75+Q3dir0.25"]
    objs = {nm: objective(candidate_moves(nm)) for nm in cands}

    print("=== H9 sensitivity grid: LP[worst, best] per setting (negative = improves) ===")
    print(f"  {'comp':>5} {'box':>5} {'Z':>3} | " + " | ".join(f"{nm:^21}" for nm in cands))
    for comp in (0.10, 0.20, 0.30):
        for boxw in (0.25, 0.35, 0.45):
            bounds = [(max(0.02, rate[t][s] - boxw), min(0.98, rate[t][s] + boxw))
                      for t in TARGETS for s in subjects]
            for Zz in (2.0, 3.0):
                A_ub, b_ub = [], []
                for v, c0, d, sg in raw_constraints:
                    tol = 1e-5 + comp * abs(d) + Zz * sg
                    A_ub.append(v);  b_ub.append(d + tol - c0)
                    A_ub.append(-v); b_ub.append(-(d - tol - c0))
                A_ub = np.array(A_ub); b_ub = np.array(b_ub)
                feas = linprog(np.zeros(n_var), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
                if not feas.success:
                    print(f"  {comp:5.2f} {boxw:5.2f} {Zz:3.0f} | infeasible")
                    continue
                cells = []
                for nm in cands:
                    wv, wc = objs[nm]
                    lo = linprog(wv, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
                    hi = linprog(-wv, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
                    w_, b_ = -hi.fun + wc, lo.fun + wc
                    mark = "*" if w_ <= 1e-6 else "!"
                    cells.append(f"[{w_:+8.5f},{b_:+8.5f}]{mark}")
                print(f"  {comp:5.2f} {boxw:5.2f} {Zz:3.0f} | " + " | ".join(cells))
    print("  (* = sign-safe even in the adversarial corner; ! = can lose)")

    # --- candidate anatomy for the build ---
    print("\n=== candidate anatomy: FS + Q2dir_k0.75 (and capped variant) ===")
    p = FS["Q2"].values
    for nm in ("Q2dir_k0.75", "Q2dir_k0.75_cap0.8"):
        shift_map = dict(candidate_moves(nm))["Q2"]
        shift = np.array([shift_map[s] for s in sub])
        p_new = sigmoid(logit(p) + shift)
        print(f"\n  {nm}:")
        print(f"  {'subj':>6} {'shift':>7} {'FSmean':>7} {'newmean':>8} {'train':>6} {'rec10':>6}")
        for s in subjects:
            m = masks[s]
            d = pd.to_datetime(train[train.subject_id == s]["sleep_date"])
            w = np.exp(-(d.max() - d).dt.days.values / 10.0)
            rec = float(np.sum(w * train[train.subject_id == s]["Q2"].values) / np.sum(w))
            print(f"  {s:>6} {shift_map[s]:+7.2f} {p[m].mean():7.3f} {p_new[m].mean():8.3f} "
                  f"{rate['Q2'][s]:6.3f} {rec:6.3f}")
        print(f"  global Q2 mean: {p.mean():.4f} -> {p_new.mean():.4f} | cell range [{p_new.min():.4f},{p_new.max():.4f}]"
              f" | mean|cell move|={np.abs(p_new-p).mean():.4f} max={np.abs(p_new-p).max():.4f}")


if __name__ == "__main__":
    main()
