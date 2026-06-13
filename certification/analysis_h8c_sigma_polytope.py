#!/usr/bin/env python3
"""H8c: sigma-calibrated label polytope + the quantitative transfer law.

KEY DERIVATION: for any move, per-cell loss spread between y=1 and y=0 equals the cell's
LOGIT displacement. So under the per-subject-rate model, the realization-noise scale of a
move's LB delta is
        sigma(move) = 0.5 * ||delta_logit||_2 / 1750        (Bernoulli var <= 1/4)
Dense uniform level moves: |expected rate-driven delta| >> sigma  -> predictable.
Sparse cell-surgery moves: |delta| << sigma                      -> row-alignment lottery.
This is the quantitative transfer law; it also dictates each ledger constraint's honest
tolerance: tol_i = rounding + z * sigma_i.

Pipeline: build all locatable ledger constraints (sigma-calibrated) + 3 old-line probes,
check feasibility, LP bounds for FS level shifts, hit-and-run posterior, S/N table.
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
Z = 2.0

FS_FILE = "submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv"


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


c1 = import_module(HERE / "candidate_1_public_loss_sparse_tomography.py", "c1_for_h8c")


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


def shift_logit(p, tau):
    p = np.clip(p, EPS, 1 - EPS)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + tau)))


def main():
    train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
    H057 = load_ledger_file("submission_h057_q2row_fullvector_state_7cde1a77_uploadsafe.csv")
    FS = load_ledger_file(FS_FILE)
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
        """Constraint coefs + ASSIGNMENT-NOISE sigma.
        THEOREM: (dl1 - dl0)_i = -dlogit_i. For per-subject-constant LOGIT moves this is
        constant within subject -> the pair delta depends only on per-subject label
        fractions (zero assignment noise). Generally the assignment noise is governed by
        the WITHIN-SUBJECT CENTERED part of dlogit:
            sigma_a = 0.5 * sqrt( sum_s sum_{i in s} (dlogit_i - mean_s dlogit)^2 ) / 1750
        """
        v = np.zeros(n_var); c0 = 0.0; sq_centered = 0.0
        for t in TARGETS:
            coef, const, dlg = lin_terms(df_from[t].values, df_to[t].values)
            c0 += const.sum() / N_CELLS
            for s in subjects:
                m = masks[s]
                v[vidx[(s, t)]] += coef[m].sum() / N_CELLS
                sq_centered += float(((dlg[m] - dlg[m].mean()) ** 2).sum())
        sigma = 0.5 * np.sqrt(sq_centered) / N_CELLS
        return v, c0, sigma

    constraints = []
    ledger = pd.read_csv(ETRI / "data_analytics" / "hsjepa_public_score_ledger.csv")
    for rec in ledger.to_dict("records"):
        df = load_ledger_file(str(rec["file"]))
        if df is None:
            continue
        v, c0, sg = pair_row(H057, df)
        d = float(rec["public_lb"]) - H057_LB
        constraints.append((str(rec["file"])[:44], v, c0, d, sg))
    olds = {kk: load_abs(OLD / nm) for kk, nm in {
        "M": "submission_lb60247_to_pertarget_best_microblend.csv",
        "A": "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv",
        "B": "submission_Q1up100_plus_Q2recency_tau10_lam075.csv",
        "C": "submission_FINAL_Q2recency_plus_Q3recency.csv"}.items()}
    for nm, (f1, f2, d, rnd) in {
        "old_Q2probe": ("A", "B", 0.5949167449 - 0.6001, 7e-5),
        "old_Q3probe": ("B", "C", 0.593188787 - 0.5949167449, 2e-6),
        "old_Q1probe": ("M", "A", 0.6001 - 0.60247, 8e-5)}.items():
        v, c0, sg = pair_row(olds[f1], olds[f2])
        constraints.append((nm, v, c0, d, sg))

    print("=== S/N table: the quantitative transfer law ===")
    print(f"  {'pair (vs H057 / probe)':>46} {'measured':>9} {'sigma':>8} {'S/N':>6}")
    for nm, _v, _c0, d, sg in constraints:
        print(f"  {nm:>46} {d:+9.5f} {sg:8.5f} {abs(d)/sg:6.2f}")
    c3p = c1.locate("submission_final_candidate3_inhull_loo_consensus_b1ebf188_uploadsafe.csv")
    if c3p:
        _v, _c, sg3 = pair_row(H057, load_abs(c3p))
        print(f"  {'candidate3 (sparse action, unsubmitted)':>46} {'?':>9} {sg3:8.5f}")
    for tau in (0.10, 0.15):
        _v, _c, sgl = pair_row(FS, FS.assign(Q2=shift_logit(FS['Q2'].values, tau)))
        print(f"  {'FS+Q2 uniform tau=' + format(tau,'.2f') + ' (level action)':>46} {'?':>9} {sgl:8.5f}")

    rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}
    bounds = [(max(0.02, rate[t][s] - 0.35), min(0.98, rate[t][s] + 0.35))
              for t in TARGETS for s in subjects]

    A_ub, b_ub = [], []
    for _nm, v, c0, d, sg in constraints:
        # tol = abs floor + public-subset composition (~10% of |d|) + Z * assignment noise
        tol = 1e-5 + 0.10 * abs(d) + Z * sg
        A_ub.append(v);  b_ub.append(d + tol - c0)
        A_ub.append(-v); b_ub.append(-(d - tol - c0))
    A_ub = np.array(A_ub); b_ub = np.array(b_ub)
    feas = linprog(np.zeros(n_var), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    print(f"\npolytope (z={Z}) feasible: {feas.success}")
    if not feas.success:
        print("  still infeasible -> increase z or inspect; stopping")
        return

    def objective_row(target, tau):
        v = np.zeros(n_var)
        p = FS[target].values
        coef, const, _ = lin_terms(p, shift_logit(p, tau))
        c0 = const.sum() / N_CELLS
        for s in subjects:
            v[vidx[(s, target)]] += coef[masks[s]].sum() / N_CELLS
        return v, c0

    print("\n=== LP bounds over sigma-polytope (rate-driven component; negative = improves) ===")
    taus = [0.05, 0.10, 0.15, 0.20, 0.30]
    print(f"  {'target':>6} {'tau':>5} | {'WORST':>9} {'BEST':>9}")
    lp_rows = {}
    for target in ["Q2", "Q1", "Q3"]:
        for tau in taus:
            ov, oc = objective_row(target, tau)
            lo = linprog(ov, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            hi = linprog(-ov, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            w_, b_ = -hi.fun + oc, lo.fun + oc
            lp_rows[(target, tau)] = (w_, b_)
            print(f"  {target:>6} {tau:5.2f} | {w_:+9.5f} {b_:+9.5f}"
                  + ("   <-- sign-safe" if w_ <= 1e-6 else ""))

    print("\n=== hit-and-run uniform posterior ===")
    # Chebyshev center with the BOX folded into the inequalities so x is strictly interior
    lob = np.array([b[0] for b in bounds]); hib = np.array([b[1] for b in bounds])
    A_all = np.vstack([A_ub, np.eye(n_var), -np.eye(n_var)])
    b_all = np.concatenate([b_ub, hib, -lob])
    A_che = np.hstack([A_all, np.ones((A_all.shape[0], 1))])
    c_che = np.zeros(n_var + 1); c_che[-1] = -1.0
    che = linprog(c_che, A_ub=A_che, b_ub=b_all,
                  bounds=[(None, None)] * n_var + [(0, None)], method="highs")
    x = che.x[:n_var]
    print(f"  chebyshev slack: {che.x[-1]:.2e}")
    rng = np.random.default_rng(11)
    samples = []
    for step in range(60_000):
        u = rng.standard_normal(n_var); u /= np.linalg.norm(u)
        t1 = np.inf; t0 = -np.inf
        pos = u > 1e-12; neg = u < -1e-12
        if pos.any():
            t1 = min(t1, ((hib - x)[pos] / u[pos]).min())
            t0 = max(t0, ((lob - x)[pos] / u[pos]).max())
        if neg.any():
            t1 = min(t1, ((lob - x)[neg] / u[neg]).min())
            t0 = max(t0, ((hib - x)[neg] / u[neg]).max())
        au = A_ub @ u; ax = A_ub @ x
        p2 = au > 1e-14; n2 = au < -1e-14
        if p2.any():
            t1 = min(t1, ((b_ub - ax)[p2] / au[p2]).min())
        if n2.any():
            t0 = max(t0, ((b_ub - ax)[n2] / au[n2]).max())
        if t1 <= t0:
            continue
        x = x + (t0 + (t1 - t0) * rng.random()) * u
        if step % 30 == 0 and step > 5000:
            samples.append(x.copy())
    S = np.array(samples)
    print(f"  samples: {len(S)}")
    print(f"  {'target':>6} {'tau':>5} | {'mean':>9} {'q05':>9} {'q50':>9} {'q95':>9} {'P(<0)':>6}")
    for target in ["Q2", "Q1", "Q3"]:
        for tau in taus:
            ov, oc = objective_row(target, tau)
            g = S @ ov + oc
            q05, q50, q95 = np.percentile(g, [5, 50, 95])
            print(f"  {target:>6} {tau:5.2f} | {g.mean():+9.5f} {q05:+9.5f} {q50:+9.5f} {q95:+9.5f} {(g<0).mean():6.2f}")

    print("\n=== joint Q1+Q2 combos (posterior; add candidate realization sigma ~0.0005 on top) ===")
    print(f"  {'tauQ2':>5} {'tauQ1':>5} | {'mean':>9} {'q05':>9} {'q95':>9} {'P(<0)':>6}")
    for tq2 in (0.0, 0.05, 0.10, 0.15):
        for tq1 in (0.0, 0.05, 0.10):
            if tq2 == 0 and tq1 == 0:
                continue
            g = np.zeros(len(S))
            if tq2 > 0:
                ov, oc = objective_row("Q2", tq2); g = g + S @ ov + oc
            if tq1 > 0:
                ov, oc = objective_row("Q1", tq1); g = g + S @ ov + oc
            q05, q95 = np.percentile(g, [5, 95])
            print(f"  {tq2:5.2f} {tq1:5.2f} | {g.mean():+9.5f} {q05:+9.5f} {q95:+9.5f} {(g<0).mean():6.2f}")


if __name__ == "__main__":
    main()
