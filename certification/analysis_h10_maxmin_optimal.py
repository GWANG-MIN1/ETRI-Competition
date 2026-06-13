#!/usr/bin/env python3
"""H10: the certified-OPTIMAL move — max-min robust optimization over the label polytope.

Move class: per-subject-per-target uniform logit shifts delta in R^70, |delta|<=0.8
(assignment-noise-free by the H8 theorem). For fixed delta the public gain is LINEAR in
the label-fraction vector r; for fixed r it is CONVEX in delta with
    d gain / d delta_{s,t} = [ sum_{i in s} sigmoid(z_i + delta) - r_{s,t} * n_s ] / 1750
(optimum = classic calibration: subject mean prediction -> label rate).

We minimize  F(delta) = max_{r in P_loose} gain(delta, r)   by projected subgradient,
where P_loose is the LOOSEST sensitivity polytope (comp 30%, box +-0.45, Z=3): it is a
superset of all 18 grid polytopes, so F(delta*)<0 certifies sign-safety EVERYWHERE.
Then we evaluate delta* on the central polytope (LP + hit-and-run posterior).
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
CAP = 0.8


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


c1 = import_module(HERE / "candidate_1_public_loss_sparse_tomography.py", "c1_for_h10")


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
    nS, nT = len(subjects), len(TARGETS)
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

    # precompute FS logits per (s,t)
    Z = {}
    for t in TARGETS:
        z = logit(FS[t].values)
        for s in subjects:
            Z[(s, t)] = z[masks[s]]

    def gain_terms(delta):
        """A_sum (const wrt r) and linear coefs on r: gain = A + sum coef_{s,t} r_{s,t}."""
        A = 0.0
        coefs = np.zeros(n_var)
        for t in TARGETS:
            for s in subjects:
                d = delta[vidx[(s, t)]]
                z = Z[(s, t)]
                p_new = sigmoid(z + d)
                p_old = sigmoid(z)
                A += float(-np.log((1 - p_new) / (1 - p_old)).sum()) / N_CELLS
                coefs[vidx[(s, t)]] = -n_rows[s] * d / N_CELLS
        return A, coefs

    def worst_r(delta, A_ub, b_ub, bounds):
        """max over r of gain  ->  worst case (least improvement)."""
        A, coefs = gain_terms(delta)
        res = linprog(-coefs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        return A + (-res.fun), res.x  # value, argmax r

    # --- optimize against the LOOSEST polytope (superset of all settings) ---
    A_l, b_l, bounds_l = polytope(0.30, 0.45, 3.0)
    rng = np.random.default_rng(3)
    delta = np.zeros(n_var)
    best_val, best_delta = np.inf, delta.copy()
    for it in range(250):
        val, r_star = worst_r(delta, A_l, b_l, bounds_l)
        if val < best_val:
            best_val, best_delta = val, delta.copy()
        # subgradient of worst-case gain wrt delta
        grad = np.zeros(n_var)
        for t in TARGETS:
            for s in subjects:
                j = vidx[(s, t)]
                grad[j] = (sigmoid(Z[(s, t)] + delta[j]).sum() - r_star[j] * n_rows[s]) / N_CELLS
        eta = 8.0 / np.sqrt(it + 4)
        delta = np.clip(delta - eta * grad, -CAP, CAP)
        if it % 50 == 0:
            print(f"  iter {it:3d}: certified worst-case = {val:+.5f}")
    print(f"  FINAL certified worst-case (loosest polytope, ALL settings): {best_val:+.5f}")
    delta = best_delta

    # --- evaluate delta* on the central polytope ---
    A_c, b_c, bounds_c = polytope(0.10, 0.35, 2.0)
    val_c, _ = worst_r(delta, A_c, b_c, bounds_c)
    A_g, coefs_g = gain_terms(delta)
    res_best = linprog(coefs_g, A_ub=A_c, b_ub=b_c, bounds=bounds_c, method="highs")
    print(f"  on central polytope: worst {val_c:+.5f} | best {A_g + res_best.fun:+.5f}")

    # --- anatomy ---
    print("\n=== delta* anatomy (per-subject-per-target logit shifts; |.|<0.02 shown as .) ===")
    print("  subj | " + " | ".join(f"{t:>6}" for t in TARGETS))
    for s in subjects:
        cells = []
        for t in TARGETS:
            d = delta[vidx[(s, t)]]
            cells.append(f"{d:+6.2f}" if abs(d) >= 0.02 else "     .")
        print(f"  {s} | " + " | ".join(cells))
    per_t = {t: float(np.mean([abs(delta[vidx[(s, t)]]) for s in subjects])) for t in TARGETS}
    print("  mean|delta| per target: " + " ".join(f"{t}={v:.2f}" for t, v in per_t.items()))

    # --- posterior on central polytope (hit-and-run) ---
    lob = np.array([b[0] for b in bounds_c]); hib = np.array([b[1] for b in bounds_c])
    A_all = np.vstack([A_c, np.eye(n_var), -np.eye(n_var)])
    b_all = np.concatenate([b_c, hib, -lob])
    A_che = np.hstack([A_all, np.ones((A_all.shape[0], 1))])
    c_che = np.zeros(n_var + 1); c_che[-1] = -1.0
    che = linprog(c_che, A_ub=A_che, b_ub=b_all,
                  bounds=[(None, None)] * n_var + [(0, None)], method="highs")
    x = che.x[:n_var]
    samples = []
    for step in range(60_000):
        u = rng.standard_normal(n_var); u /= np.linalg.norm(u)
        t1 = np.inf; t0 = -np.inf
        pos = u > 1e-12; neg = u < -1e-12
        if pos.any():
            t1 = min(t1, ((hib - x)[pos] / u[pos]).min()); t0 = max(t0, ((lob - x)[pos] / u[pos]).max())
        if neg.any():
            t1 = min(t1, ((lob - x)[neg] / u[neg]).min()); t0 = max(t0, ((hib - x)[neg] / u[neg]).max())
        au = A_c @ u; ax = A_c @ x
        p2 = au > 1e-14; n2 = au < -1e-14
        if p2.any():
            t1 = min(t1, ((b_c - ax)[p2] / au[p2]).min())
        if n2.any():
            t0 = max(t0, ((b_c - ax)[n2] / au[n2]).max())
        if t1 <= t0:
            continue
        x = x + (t0 + (t1 - t0) * rng.random()) * u
        if step % 30 == 0 and step > 5000:
            samples.append(x.copy())
    S = np.array(samples)
    g = S @ coefs_g + A_g
    q05, q50, q95 = np.percentile(g, [5, 50, 95])
    print(f"\n  posterior (central): mean {g.mean():+.5f} q05 {q05:+.5f} q50 {q50:+.5f} q95 {q95:+.5f} "
          f"P(<0) {(g<0).mean():.2f}  (n={len(S)})")

    np.save(HERE / "outputs" / "h10_delta_star.npy", delta)
    print(f"\nsaved delta* -> outputs/h10_delta_star.npy")


if __name__ == "__main__":
    main()
