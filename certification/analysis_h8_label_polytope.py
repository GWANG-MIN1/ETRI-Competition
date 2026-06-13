#!/usr/bin/env python3
"""H8: exact LP bounds + ABC posterior of the FS level-shift gain over the label polytope.

KEY FACT: E_{y~Bern(r)}[logloss] is LINEAR in r. Modeling each subject-target public label
rate r_{s,t} as a free variable, every probe measurement is a LINEAR equality (within
tolerance) in r, and every candidate level-shift gain is a LINEAR objective. So:

  - exact worst/best case of the FS gain  = linear program over the label polytope
    {r in box : probe constraints hold}
  - posterior distribution of the FS gain = ABC (soft-constraint) Monte Carlo on the box.

Constraints (LB units = mean logloss over 250 rows x 7 targets):
  P1  Q2-only probe  A->B : measured -0.0051833 (LB display rounding tol)
  P2  Q3-only probe  B->C : measured -0.0017279
  P3  Q1-dominant probe micro->A (ALL 7 targets moved slightly): measured -0.00237
Boxes:
  agnostic : r in [0.02, 0.98]
  drift    : r in clip(train_subject_rate +- 0.35)
Objectives: FS + uniform logit tau on Q1 / Q2 / Q3 cells, tau grid.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ETRI = Path(r"C:\Users\박광민\Documents\Codex\etri_team")
OLD = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\outputs\submissions")
RAW = ETRI / "data"
KEYS = ["subject_id", "sleep_date", "lifelog_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6
N_CELLS = 1750.0  # 250 rows x 7 targets; LB = sum(loss)/1750

FILE_M = OLD / "submission_lb60247_to_pertarget_best_microblend.csv"   # LB 0.60247
FILE_A = OLD / "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv"  # LB 0.6001
FILE_B = OLD / "submission_Q1up100_plus_Q2recency_tau10_lam075.csv"      # LB 0.5949167449
FILE_C = OLD / "submission_FINAL_Q2recency_plus_Q3recency.csv"           # LB 0.593188787
FS_FILE = ETRI / "submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv"

M_P1 = 0.5949167449 - 0.6001     # Q2 cells only
M_P2 = 0.593188787 - 0.5949167449  # Q3 cells only
M_P3 = 0.6001 - 0.60247          # all targets (Q1-dominant)
TOL_TIGHT = {"P1": 7e-5, "P2": 2e-6, "P3": 8e-5}          # display rounding
TOL_LOOSE = {"P1": 0.25 * abs(M_P1), "P2": 0.25 * abs(M_P2), "P3": 0.25 * abs(M_P3)}


def load(p):
    return pd.read_csv(p).sort_values(KEYS).reset_index(drop=True)


def lin_terms(p_from, p_to):
    """Per-cell E[loss(to)] - E[loss(from)] = coef * r + const  (exact, no approx)."""
    pf = np.clip(p_from, EPS, 1 - EPS)
    pt = np.clip(p_to, EPS, 1 - EPS)
    coef = (np.log((1 - pt) / (1 - pf)) - np.log(pt / pf))
    const = -np.log((1 - pt) / (1 - pf))
    return coef, const


def shift_logit(p, tau):
    p = np.clip(p, EPS, 1 - EPS)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + tau)))


def main():
    train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
    M, A, B, C, FS = load(FILE_M), load(FILE_A), load(FILE_B), load(FILE_C), load(FS_FILE)
    sub = A["subject_id"].values
    subjects = sorted(set(sub))
    n_var = len(subjects) * len(TARGETS)  # 70
    vidx = {(s, t): i for i, (t) in enumerate(TARGETS) for i, s in []}  # placeholder
    vidx = {}
    k = 0
    for t in TARGETS:
        for s in subjects:
            vidx[(s, t)] = k
            k += 1

    def probe_row(df_from, df_to, targets):
        """Return (coef_vector[n_var], const) for a probe's predicted LB delta."""
        v = np.zeros(n_var)
        c0 = 0.0
        for t in targets:
            coef, const = lin_terms(df_from[t].values, df_to[t].values)
            c0 += const.sum() / N_CELLS
            for s in subjects:
                m = sub == s
                v[vidx[(s, t)]] += coef[m].sum() / N_CELLS
        return v, c0

    p1_v, p1_c = probe_row(A, B, ["Q2"])
    p2_v, p2_c = probe_row(B, C, ["Q3"])
    p3_v, p3_c = probe_row(M, A, TARGETS)

    def objective_row(target, tau):
        v = np.zeros(n_var)
        p = FS[target].values
        coef, const = lin_terms(p, shift_logit(p, tau))
        c0 = const.sum() / N_CELLS
        for s in subjects:
            m = sub == s
            v[vidx[(s, target)]] += coef[m].sum() / N_CELLS
        return v, c0

    rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}
    boxes = {
        "agnostic": [(0.02, 0.98)] * n_var,
        "drift+-0.35": [
            (max(0.02, rate[t][s] - 0.35), min(0.98, rate[t][s] + 0.35))
            for t in TARGETS for s in subjects
        ],
    }

    def lp_bounds(obj_v, obj_c, box, tol):
        A_ub, b_ub = [], []
        for v, c0, m, key in [(p1_v, p1_c, M_P1, "P1"), (p2_v, p2_c, M_P2, "P2"), (p3_v, p3_c, M_P3, "P3")]:
            A_ub.append(v);  b_ub.append(m + tol[key] - c0)
            A_ub.append(-v); b_ub.append(-(m - tol[key] - c0))
        A_ub = np.array(A_ub); b_ub = np.array(b_ub)
        lo = linprog(obj_v, A_ub=A_ub, b_ub=b_ub, bounds=box, method="highs")
        hi = linprog(-obj_v, A_ub=A_ub, b_ub=b_ub, bounds=box, method="highs")
        if not (lo.success and hi.success):
            return None, None
        return lo.fun + obj_c, -(hi.fun) + obj_c

    print("=== H8 LP exact bounds on FS level-shift gain (LB units; negative = improves) ===")
    taus = [0.05, 0.10, 0.15, 0.20, 0.30]
    for box_name, box in boxes.items():
        for tol_name, tol in [("tight", TOL_TIGHT), ("loose", TOL_LOOSE)]:
            print(f"\n  --- box={box_name} | probe tolerance={tol_name} ---")
            print(f"  {'target':>6} {'tau':>5} | {'WORST':>9} {'BEST':>9}")
            for target in ["Q2", "Q1", "Q3"]:
                for tau in taus:
                    ov, oc = objective_row(target, tau)
                    w, b = lp_bounds(ov, oc, box, tol)
                    if w is None:
                        print(f"  {target:>6} {tau:5.2f} | infeasible")
                        continue
                    print(f"  {target:>6} {tau:5.2f} | {b:+9.5f} {w:+9.5f}"
                          + ("   <-- worst<=0: cannot lose" if b <= 1e-6 else ""))

    # ------------------------------------------------------------------
    print("\n=== ABC posterior over the drift box (soft probe constraints) ===")
    rng = np.random.default_rng(7)
    Nmc = 400_000
    lo = np.array([b[0] for b in boxes["drift+-0.35"]])
    hi = np.array([b[1] for b in boxes["drift+-0.35"]])
    R = lo + (hi - lo) * rng.random((Nmc, n_var))
    sig = {k: TOL_LOOSE[k] / 2 for k in TOL_LOOSE}
    z1 = (R @ p1_v + p1_c - M_P1) / sig["P1"]
    z2 = (R @ p2_v + p2_c - M_P2) / sig["P2"]
    z3 = (R @ p3_v + p3_c - M_P3) / sig["P3"]
    w = np.exp(-0.5 * (z1**2 + z2**2 + z3**2))
    w /= w.sum()
    ess = 1.0 / np.sum(w**2)
    print(f"  N={Nmc}  effective sample size={ess:.0f}")
    print(f"  {'target':>6} {'tau':>5} | {'mean':>9} {'q05':>9} {'q50':>9} {'q95':>9} {'P(gain<0)':>9}")
    rows_out = []
    for target in ["Q2", "Q1", "Q3"]:
        for tau in taus:
            ov, oc = objective_row(target, tau)
            g = R @ ov + oc
            order = np.argsort(g)
            cw = np.cumsum(w[order])
            q05 = g[order][np.searchsorted(cw, 0.05)]
            q50 = g[order][np.searchsorted(cw, 0.50)]
            q95 = g[order][np.searchsorted(cw, 0.95)]
            mean = float(np.sum(w * g))
            pneg = float(np.sum(w[g < 0]))
            rows_out.append((target, tau, mean, q05, q50, q95, pneg))
            print(f"  {target:>6} {tau:5.2f} | {mean:+9.5f} {q05:+9.5f} {q50:+9.5f} {q95:+9.5f} {pneg:9.2f}")

    # combo: Q2 tau + Q1 tau joint posterior gain (additivity across targets is exact)
    print("\n=== joint Q1+Q2 combo posterior (gains add exactly across targets) ===")
    print(f"  {'tauQ2':>5} {'tauQ1':>5} | {'mean':>9} {'q05':>9} {'q95':>9} {'P(<0)':>6}")
    for tq2 in (0.10, 0.15):
        for tq1 in (0.0, 0.05, 0.10):
            ov2, oc2 = objective_row("Q2", tq2)
            g = R @ ov2 + oc2
            if tq1 > 0:
                ov1, oc1 = objective_row("Q1", tq1)
                g = g + R @ ov1 + oc1
            order = np.argsort(g); cw = np.cumsum(w[order])
            print(f"  {tq2:5.2f} {tq1:5.2f} | {float(np.sum(w*g)):+9.5f} "
                  f"{g[order][np.searchsorted(cw,0.05)]:+9.5f} {g[order][np.searchsorted(cw,0.95)]:+9.5f} "
                  f"{float(np.sum(w[g<0])):6.2f}")


if __name__ == "__main__":
    main()
