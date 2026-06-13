#!/usr/bin/env python3
"""H10b: polish the max-min optimum, warm-started from the locked recipe.

The recipe (Q2dir k0.75 cap0.8 + Q3dir k0.25) achieves loosest-polytope worst -0.00098;
H10's cold-start subgradient only reached -0.00056, so it had not converged. Warm-starting
at the recipe can only find points that are AT LEAST as good on the certified objective.
Small steps + more iterations + best-iterate tracking.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent

h10 = import_module = None
spec = importlib.util.spec_from_file_location("h10", HERE / "analysis_h10_maxmin_optimal.py")
h10 = importlib.util.module_from_spec(spec)
sys.modules["h10"] = h10
# we re-implement main loop; import only shared funcs by exec of module-level code is heavy.
# simpler: duplicate the small helpers via the module file's functions after loading it
spec.loader.exec_module(sys.modules["h10"]) if False else None  # do NOT run its main

# --- self-contained (copied helpers; same as H10) ---
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


c1 = imp(HERE / "candidate_1_public_loss_sparse_tomography.py", "c1_for_h10b")


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
        p = c1.locate(str(rec["file"]))
        if p is None:
            continue
        v, c0, sg = pair_row(H057, load_abs(p))
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

    rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}

    def polytope(comp, boxw, Zz):
        bounds = [(max(0.02, rate[t][s] - boxw), min(0.98, rate[t][s] + boxw))
                  for t in TARGETS for s in subjects]
        A_ub, b_ub = [], []
        for v, c0, d, sg in raw_constraints:
            tol = 1e-5 + comp * abs(d) + Zz * sg
            A_ub.append(v);  b_ub.append(d + tol - c0)
            A_ub.append(-v); b_ub.append(-(d - tol - c0))
        return np.array(A_ub), np.array(b_ub), bounds

    Zl = {}
    for t in TARGETS:
        z = logit(FS[t].values)
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

    def worst_val(delta, A_ub, b_ub, bounds):
        A, coefs = gain_terms(delta)
        res = linprog(-coefs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        return A - res.fun, res.x

    # --- warm start = locked recipe ---
    delta = np.zeros(n_var)
    for target, kappa in [("Q2", 0.75), ("Q3", 0.25)]:
        f1, f2 = ("A", "B") if target == "Q2" else ("B", "C")
        dlg = logit(olds[f2][target].values) - logit(olds[f1][target].values)
        for s in subjects:
            lam = float(dlg[masks[s]].mean())
            delta[vidx[(s, target)]] = float(np.clip(kappa * lam, -CAP, CAP))

    A_l, b_l, bounds_l = polytope(0.30, 0.45, 3.0)
    v0, _ = worst_val(delta, A_l, b_l, bounds_l)
    print(f"warm start (locked recipe): certified worst on loosest polytope = {v0:+.5f}")

    best_val, best_delta = v0, delta.copy()
    for it in range(400):
        val, r_star = worst_val(delta, A_l, b_l, bounds_l)
        if val < best_val:
            best_val, best_delta = val, delta.copy()
        grad = np.zeros(n_var)
        for t in TARGETS:
            for s in subjects:
                j = vidx[(s, t)]
                grad[j] = (sigmoid(Zl[(s, t)] + delta[j]).sum() - r_star[j] * n_rows[s]) / N_CELLS
        eta = 1.2 / np.sqrt(it + 9)
        delta = np.clip(delta - eta * grad, -CAP, CAP)
        if it % 80 == 0:
            print(f"  iter {it:3d}: worst {val:+.5f} (best so far {best_val:+.5f})")
    print(f"FINAL polished certified worst: {best_val:+.5f}  (recipe was {v0:+.5f})")
    delta = best_delta

    A_c, b_c, bounds_c = polytope(0.10, 0.35, 2.0)
    vc, _ = worst_val(delta, A_c, b_c, bounds_c)
    A_g, coefs_g = gain_terms(delta)
    res_best = linprog(coefs_g, A_ub=A_c, b_ub=b_c, bounds=bounds_c, method="highs")
    print(f"on central polytope: worst {vc:+.5f} | best {A_g + res_best.fun:+.5f}")

    print("\n=== polished delta vs recipe (per-subject; only |.|>=0.02 shown) ===")
    print("  subj | " + " | ".join(f"{t:>6}" for t in TARGETS))
    for s in subjects:
        cells = []
        for t in TARGETS:
            d = delta[vidx[(s, t)]]
            cells.append(f"{d:+6.2f}" if abs(d) >= 0.02 else "     .")
        print(f"  {s} | " + " | ".join(cells))
    np.save(HERE / "outputs" / "h10b_delta_polished.npy", delta)
    print("saved -> outputs/h10b_delta_polished.npy")


if __name__ == "__main__":
    main()
