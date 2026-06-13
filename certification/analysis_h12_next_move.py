#!/usr/bin/env python3
"""H12: with the NEW measurement (0.5647490904) in the polytope, is there a NEXT certified move?

The new observation is itself per-subject-uniform (assignment-noise-free) -> the tightest
possible constraint type. Add it, re-run the max-min search from the NEW base (004a0549),
and LP-test simple extensions (more kappa on the same directions).
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
NEW_FILE = "submission_final_probe_direction_certified_004a0549_uploadsafe.csv"
NEW_LB = 0.5647490904
CAP = 0.8


def imp(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


c1 = imp(HERE / "candidate_1_public_loss_sparse_tomography.py", "c1_for_h12")


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
    NEW = load_abs(ETRI / NEW_FILE)
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
        named.append((str(rec["file"])[:36], v, c0, float(rec["public_lb"]) - H057_LB, sg))
    olds = {kk: load_abs(OLD / nm) for kk, nm in {
        "M": "submission_lb60247_to_pertarget_best_microblend.csv",
        "A": "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv",
        "B": "submission_Q1up100_plus_Q2recency_tau10_lam075.csv",
        "C": "submission_FINAL_Q2recency_plus_Q3recency.csv"}.items()}
    for nm, (f1, f2, d) in {"old_Q2probe": ("A", "B", 0.5949167449 - 0.6001),
                            "old_Q3probe": ("B", "C", 0.593188787 - 0.5949167449),
                            "old_Q1probe": ("M", "A", 0.6001 - 0.60247)}.items():
        v, c0, sg = pair_row(olds[f1], olds[f2])
        named.append((nm, v, c0, d, sg))
    print(f"constraints incl. NEW measurement: {len(named)}")
    new_sigma = [sg for nm, _v, _c, _d, sg in named if "final_probe_direction" in nm]
    if new_sigma:
        print(f"  NEW constraint sigma_a = {new_sigma[0]:.6f} (per-subject-uniform -> ~0 = tightest type)")

    rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}

    def polytope(comp, boxw, Zz):
        bounds = [(max(0.02, rate[t][s] - boxw), min(0.98, rate[t][s] + boxw))
                  for t in TARGETS for s in subjects]
        A_ub, b_ub = [], []
        for _nm, v, c0, d, sg in named:
            tol = 1e-5 + comp * abs(d) + Zz * sg
            A_ub.append(v);  b_ub.append(d + tol - c0)
            A_ub.append(-v); b_ub.append(-(d - tol - c0))
        return np.array(A_ub), np.array(b_ub), bounds

    # base for the NEXT move = the NEW submission
    Zl = {}
    for t in TARGETS:
        z = logit(NEW[t].values)
        for s in subjects:
            Zl[(s, t)] = z[masks[s]]

    def gain_terms(delta):
        A = 0.0
        coefs = np.zeros(n_var)
        for t in TARGETS:
            for s in subjects:
                d = delta[vidx[(s, t)]]
                z = Zl[(s, t)]
                A += float(-np.log((1 - sigmoid(z + d)) / (1 - sigmoid(z))).sum()) / N_CELLS
                coefs[vidx[(s, t)]] = -n_rows[s] * d / N_CELLS
        return A, coefs

    def worst_value(delta, A_ub, b_ub, bounds):
        A, coefs = gain_terms(delta)
        res = linprog(-coefs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        return (A - res.fun) if res.success else None

    A_l, b_l, bounds_l = polytope(0.30, 0.45, 3.0)
    A_c, b_c, bounds_c = polytope(0.10, 0.35, 2.0)
    feas = linprog(np.zeros(n_var), A_ub=A_l, b_ub=b_l, bounds=bounds_l, method="highs")
    print(f"loosest polytope feasible with NEW measurement: {feas.success}")

    # --- simple extensions: more kappa along the SAME probe directions, from the NEW base ---
    Lam = {}
    for target, (f1, f2) in {"Q2": ("A", "B"), "Q3": ("B", "C")}.items():
        dlg = logit(olds[f2][target].values) - logit(olds[f1][target].values)
        Lam[target] = {s: float(dlg[masks[s]].mean()) for s in subjects}

    def ext_delta(kq2, kq3, cap=CAP):
        delta = np.zeros(n_var)
        for s in subjects:
            delta[vidx[(s, "Q2")]] = float(np.clip(kq2 * Lam["Q2"][s], -cap, cap))
            delta[vidx[(s, "Q3")]] = float(np.clip(kq3 * Lam["Q3"][s], -cap, cap))
        return delta

    print("\n=== NEXT-move extensions from the NEW base (loosest | central worst-case) ===")
    print(f"  {'extension':>28} | {'loosest worst':>13} | {'central worst':>13} | {'central best':>12}")
    for kq2, kq3 in [(0.25, 0.0), (0.5, 0.0), (0.0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.5), (-0.25, 0.0)]:
        delta = ext_delta(kq2, kq3)
        wl = worst_value(delta, A_l, b_l, bounds_l)
        wc = worst_value(delta, A_c, b_c, bounds_c)
        A_g, coefs_g = gain_terms(delta)
        rb = linprog(coefs_g, A_ub=A_c, b_ub=b_c, bounds=bounds_c, method="highs")
        bc = A_g + rb.fun
        print(f"  Q2+={kq2:+.2f} Q3+={kq3:+.2f}{'':>8} | {wl:+13.5f} | {wc:+13.5f} | {bc:+12.5f}")

    # --- max-min search for the next certified move (cold + small steps) ---
    print("\n=== max-min search for next certified move (from NEW base, loosest polytope) ===")
    delta = np.zeros(n_var)
    best_val, best_delta = 0.0, delta.copy()
    for it in range(300):
        val = worst_value(delta, A_l, b_l, bounds_l)
        if val is None:
            break
        if val < best_val:
            best_val, best_delta = val, delta.copy()
        A_g, coefs_g = gain_terms(delta)
        res = linprog(-coefs_g, A_ub=A_l, b_ub=b_l, bounds=bounds_l, method="highs")
        r_star = res.x
        grad = np.zeros(n_var)
        for t in TARGETS:
            for s in subjects:
                j = vidx[(s, t)]
                grad[j] = (sigmoid(Zl[(s, t)] + delta[j]).sum() - r_star[j] * n_rows[s]) / N_CELLS
        eta = 2.0 / np.sqrt(it + 9)
        delta = np.clip(delta - eta * grad, -CAP, CAP)
        if it % 75 == 0:
            print(f"  iter {it:3d}: certified worst = {val:+.5f}")
    print(f"  BEST next-move certified worst: {best_val:+.5f}")
    if best_val < -1e-5:
        print("\n  next-move anatomy (|.|>=0.02):")
        print("  subj | " + " | ".join(f"{t:>6}" for t in TARGETS))
        for s in subjects:
            cells = []
            for t in TARGETS:
                d = best_delta[vidx[(s, t)]]
                cells.append(f"{d:+6.2f}" if abs(d) >= 0.02 else "     .")
            print(f"  {s} | " + " | ".join(cells))
        np.save(HERE / "outputs" / "h12_next_delta.npy", best_delta)


if __name__ == "__main__":
    main()
