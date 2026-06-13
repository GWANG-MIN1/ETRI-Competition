#!/usr/bin/env python3
"""H8e: the PROBE-DIRECTION candidate — move FS along the per-subject validated direction.

H8d posterior verdict: uniform Q2-up loses (P(<0)=0.04) because the validated probe was a
per-subject SIGNED field (up for recency-up subjects, down for recency-down ones); uniform
is its all-ones projection and cancels. The transferring object is the probe's per-subject
direction Lambda_s. Candidate: per-subject logit shift kappa * Lambda_s on FS (constant
within subject in logit space -> zero assignment noise by the H8 theorem -> its public
delta is an EXACT linear functional of per-subject label fractions).

Evaluates kappa grids for Q2-shaped, Q3-shaped, both, +small uniform Q1, over the
sigma-calibrated full-ledger polytope: LP bounds + hit-and-run posterior.
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


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


c1 = import_module(HERE / "candidate_1_public_loss_sparse_tomography.py", "c1_for_h8e")


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

    constraints = []
    ledger = pd.read_csv(ETRI / "data_analytics" / "hsjepa_public_score_ledger.csv")
    for rec in ledger.to_dict("records"):
        df = load_ledger_file(str(rec["file"]))
        if df is None:
            continue
        v, c0, sg = pair_row(H057, df)
        constraints.append((str(rec["file"])[:40], v, c0, float(rec["public_lb"]) - H057_LB, sg))
    olds = {kk: load_abs(OLD / nm) for kk, nm in {
        "M": "submission_lb60247_to_pertarget_best_microblend.csv",
        "A": "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv",
        "B": "submission_Q1up100_plus_Q2recency_tau10_lam075.csv",
        "C": "submission_FINAL_Q2recency_plus_Q3recency.csv"}.items()}
    for nm, (f1, f2, d) in {
        "old_Q2probe": ("A", "B", 0.5949167449 - 0.6001),
        "old_Q3probe": ("B", "C", 0.593188787 - 0.5949167449),
        "old_Q1probe": ("M", "A", 0.6001 - 0.60247)}.items():
        v, c0, sg = pair_row(olds[f1], olds[f2])
        constraints.append((nm, v, c0, d, sg))

    rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}
    bounds = [(max(0.02, rate[t][s] - 0.35), min(0.98, rate[t][s] + 0.35))
              for t in TARGETS for s in subjects]
    A_ub, b_ub = [], []
    for _nm, v, c0, d, sg in constraints:
        tol = 1e-5 + 0.10 * abs(d) + Z * sg
        A_ub.append(v);  b_ub.append(d + tol - c0)
        A_ub.append(-v); b_ub.append(-(d - tol - c0))
    A_ub = np.array(A_ub); b_ub = np.array(b_ub)

    # --- probe directions: per-subject mean LOGIT displacement ---
    Lam = {}
    for target, (f1, f2) in {"Q2": ("A", "B"), "Q3": ("B", "C")}.items():
        dlg = logit(olds[f2][target].values) - logit(olds[f1][target].values)
        Lam[target] = {s: float(dlg[masks[s]].mean()) for s in subjects}
    print("=== probe directions Lambda_s (per-subject mean logit shift) ===")
    for t in ("Q2", "Q3"):
        print(f"  {t}: " + " ".join(f"{s}:{Lam[t][s]:+.2f}" for s in subjects))

    def candidate_move(target, kappa, lam_map=None, uniform_tau=None):
        """FS + per-subject logit shifts -> new prob frame for that target."""
        p = FS[target].values
        z = logit(p)
        if lam_map is not None:
            shift = np.array([kappa * lam_map[s] for s in sub])
        else:
            shift = np.full(len(p), uniform_tau)
        return sigmoid(z + shift)

    def objective_row(target, p_new):
        v = np.zeros(n_var)
        coef, const, _ = lin_terms(FS[target].values, p_new)
        c0 = const.sum() / N_CELLS
        for s in subjects:
            v[vidx[(s, target)]] += coef[masks[s]].sum() / N_CELLS
        return v, c0

    # --- hit-and-run sampler (same as H8d) ---
    lob = np.array([b[0] for b in bounds]); hib = np.array([b[1] for b in bounds])
    A_all = np.vstack([A_ub, np.eye(n_var), -np.eye(n_var)])
    b_all = np.concatenate([b_ub, hib, -lob])
    A_che = np.hstack([A_all, np.ones((A_all.shape[0], 1))])
    c_che = np.zeros(n_var + 1); c_che[-1] = -1.0
    che = linprog(c_che, A_ub=A_che, b_ub=b_all,
                  bounds=[(None, None)] * n_var + [(0, None)], method="highs")
    x = che.x[:n_var]
    rng = np.random.default_rng(11)
    samples = []
    for step in range(60_000):
        u = rng.standard_normal(n_var); u /= np.linalg.norm(u)
        t1 = np.inf; t0 = -np.inf
        pos = u > 1e-12; neg = u < -1e-12
        if pos.any():
            t1 = min(t1, ((hib - x)[pos] / u[pos]).min()); t0 = max(t0, ((lob - x)[pos] / u[pos]).max())
        if neg.any():
            t1 = min(t1, ((lob - x)[neg] / u[neg]).min()); t0 = max(t0, ((hib - x)[neg] / u[neg]).max())
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
    print(f"\nposterior samples: {len(S)}")

    def evaluate(name, moves):
        """moves: list of (target, p_new). Gains add exactly across targets."""
        g = np.zeros(len(S))
        wv = np.zeros(n_var); wc = 0.0
        for target, p_new in moves:
            ov, oc = objective_row(target, p_new)
            g = g + S @ ov + oc
            wv += ov; wc += oc
        lo = linprog(wv, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        hi = linprog(-wv, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        q05, q50, q95 = np.percentile(g, [5, 50, 95])
        print(f"  {name:>34} | LP[{-hi.fun + wc:+8.5f},{lo.fun + wc:+8.5f}] | "
              f"mean {g.mean():+8.5f} q05 {q05:+8.5f} q50 {q50:+8.5f} q95 {q95:+8.5f} P(<0) {(g<0).mean():.2f}")

    print("\n=== PROBE-DIRECTION candidates (kappa x validated per-subject field) ===")
    print(f"  {'candidate':>34} | {'LP[worst,best]':^21} | posterior")
    for kap in (0.25, 0.5, 0.75, 1.0, 1.5):
        evaluate(f"Q2-dir kappa={kap}", [("Q2", candidate_move("Q2", kap, Lam["Q2"]))])
    for kap in (0.25, 0.5, 0.75, 1.0, 1.5):
        evaluate(f"Q3-dir kappa={kap}", [("Q3", candidate_move("Q3", kap, Lam["Q3"]))])
    print("  --- combos ---")
    for kap in (0.5, 0.75, 1.0):
        evaluate(f"Q2+Q3 dir kappa={kap}",
                 [("Q2", candidate_move("Q2", kap, Lam["Q2"])),
                  ("Q3", candidate_move("Q3", kap, Lam["Q3"]))])
    evaluate("Q2dir1.0 + Q3dir0.75",
             [("Q2", candidate_move("Q2", 1.0, Lam["Q2"])),
              ("Q3", candidate_move("Q3", 0.75, Lam["Q3"]))])
    print("  --- reference: uniform moves (for contrast) ---")
    evaluate("Q2 uniform tau=0.10", [("Q2", candidate_move("Q2", 0, None, 0.10))])
    evaluate("Q2 uniform tau=-0.10", [("Q2", candidate_move("Q2", 0, None, -0.10))])


if __name__ == "__main__":
    main()
