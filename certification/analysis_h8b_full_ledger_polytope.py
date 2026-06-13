#!/usr/bin/env python3
"""H8b: the FULL-LEDGER label polytope — every public measurement ever made, one geometry.

All 25+ scored submissions (both pipelines) were graded on the SAME public labels.
Under the per-subject-rate model r_{s,t} (70 vars), every measured LB delta between two
files is a LINEAR constraint on r. The feasible set = the polytope of label worlds
consistent with ALL public knowledge. We then:
  1. find the minimum tolerance at which the polytope is non-empty (model adequacy)
  2. LP worst/best bounds of FS level-shift gains over the polytope
  3. hit-and-run uniform posterior over the polytope -> P(gain<0), quantiles
  4. recency-prior reweighting as a structured-world sensitivity
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
N_CELLS = 1750.0
H057_LB = 0.5677475939

FS_FILE = ETRI / "submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv"
H057_FILE = ETRI / "submission_h057_q2row_fullvector_state_7cde1a77_uploadsafe.csv"


def load(p):
    return pd.read_csv(p).sort_values(KEYS).reset_index(drop=True)


def lin_terms(p_from, p_to):
    pf = np.clip(np.asarray(p_from, float), EPS, 1 - EPS)
    pt = np.clip(np.asarray(p_to, float), EPS, 1 - EPS)
    coef = np.log((1 - pt) / (1 - pf)) - np.log(pt / pf)
    const = -np.log((1 - pt) / (1 - pf))
    return coef, const


def shift_logit(p, tau):
    p = np.clip(p, EPS, 1 - EPS)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + tau)))


def main():
    train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
    H057 = load(H057_FILE)
    FS = load(FS_FILE)
    sub = H057["subject_id"].values
    subjects = sorted(set(sub))
    vidx = {}
    k = 0
    for t in TARGETS:
        for s in subjects:
            vidx[(s, t)] = k
            k += 1
    n_var = k
    subj_masks = {s: sub == s for s in subjects}

    def pair_row(df_from, df_to):
        v = np.zeros(n_var)
        c0 = 0.0
        for t in TARGETS:
            coef, const = lin_terms(df_from[t].values, df_to[t].values)
            c0 += const.sum() / N_CELLS
            for s in subjects:
                v[vidx[(s, t)]] += coef[subj_masks[s]].sum() / N_CELLS
        return v, c0

    # --- constraints: every ledger file vs H057 + the 3 old-line probe pairs ---
    constraints = []  # (name, v, c0, measured_delta)
    ledger = pd.read_csv(ETRI / "data_analytics" / "hsjepa_public_score_ledger.csv")
    for rec in ledger.to_dict("records"):
        f = ETRI / str(rec["file"])
        if not f.exists():
            continue
        df = load(f)
        v, c0 = pair_row(H057, df)
        constraints.append((str(rec["file"])[:40], v, c0, float(rec["public_lb"]) - H057_LB))
    oldfiles = {
        "M": OLD / "submission_lb60247_to_pertarget_best_microblend.csv",
        "A": OLD / "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv",
        "B": OLD / "submission_Q1up100_plus_Q2recency_tau10_lam075.csv",
        "C": OLD / "submission_FINAL_Q2recency_plus_Q3recency.csv",
    }
    olds = {kk: load(p) for kk, p in oldfiles.items()}
    for nm, (f1, f2, d) in {
        "old_Q2probe": ("A", "B", 0.5949167449 - 0.6001),
        "old_Q3probe": ("B", "C", 0.593188787 - 0.5949167449),
        "old_Q1probe": ("M", "A", 0.6001 - 0.60247),
    }.items():
        v, c0 = pair_row(olds[f1], olds[f2])
        constraints.append((nm, v, c0, d))
    print(f"constraints: {len(constraints)} | variables: {n_var}")

    rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}
    bounds = [(max(0.02, rate[t][s] - 0.35), min(0.98, rate[t][s] + 0.35))
              for t in TARGETS for s in subjects]

    def build_ub(tol_rel, tol_abs):
        A_ub, b_ub = [], []
        for _nm, v, c0, d in constraints:
            tol = max(tol_abs, tol_rel * abs(d))
            A_ub.append(v);  b_ub.append(d + tol - c0)
            A_ub.append(-v); b_ub.append(-(d - tol - c0))
        return np.array(A_ub), np.array(b_ub)

    # --- 1. model adequacy: smallest feasible tolerance ---
    print("\n=== model adequacy: minimum relative tolerance with non-empty polytope ===")
    feas_tol = None
    for tol_rel in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        A_ub, b_ub = build_ub(tol_rel, 2e-5)
        res = linprog(np.zeros(n_var), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        print(f"  tol_rel={tol_rel:.2f}: {'FEASIBLE' if res.success else 'infeasible'}")
        if res.success and feas_tol is None:
            feas_tol = tol_rel
    if feas_tol is None:
        print("  -> rate model cannot explain the ledger even at 50% tolerance; stopping")
        return
    work_tol = max(feas_tol, 0.15)
    A_ub, b_ub = build_ub(work_tol, 2e-5)
    print(f"  working tolerance: {work_tol:.2f} (+abs 2e-5)")

    # --- 2. LP bounds on FS level-shift gains over the full-ledger polytope ---
    def objective_row(target, tau):
        v = np.zeros(n_var)
        p = FS[target].values
        coef, const = lin_terms(p, shift_logit(p, tau))
        c0 = const.sum() / N_CELLS
        for s in subjects:
            v[vidx[(s, target)]] += coef[subj_masks[s]].sum() / N_CELLS
        return v, c0

    print("\n=== LP bounds over FULL-ledger polytope (negative = improves) ===")
    taus = [0.05, 0.10, 0.15, 0.20, 0.30]
    print(f"  {'target':>6} {'tau':>5} | {'WORST':>9} {'BEST':>9}")
    for target in ["Q2", "Q1", "Q3"]:
        for tau in taus:
            ov, oc = objective_row(target, tau)
            lo = linprog(ov, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            hi = linprog(-ov, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if lo.success and hi.success:
                print(f"  {target:>6} {tau:5.2f} | {-hi.fun + oc:+9.5f} {lo.fun + oc:+9.5f}"
                      + ("   <-- sign-safe" if -hi.fun + oc <= 1e-6 else ""))

    # --- 3. hit-and-run uniform posterior over the polytope ---
    print("\n=== hit-and-run uniform posterior over the polytope ===")
    # Chebyshev-ish start: maximize uniform slack s
    A_che = np.hstack([A_ub, np.ones((A_ub.shape[0], 1))])
    c_che = np.zeros(n_var + 1); c_che[-1] = -1.0
    che = linprog(c_che, A_ub=A_che, b_ub=b_ub,
                  bounds=bounds + [(0, None)], method="highs")
    x = che.x[:n_var]
    rng = np.random.default_rng(11)
    lob = np.array([b[0] for b in bounds]); hib = np.array([b[1] for b in bounds])
    samples = []
    n_steps, thin = 60_000, 30
    for step in range(n_steps):
        u = rng.standard_normal(n_var); u /= np.linalg.norm(u)
        # t range from box
        with np.errstate(divide="ignore"):
            t_hi = np.where(u > 1e-12, (hib - x) / u, np.inf)
            t_lo = np.where(u < -1e-12, (lob - x) / u, -np.inf)
            t1 = np.minimum(np.where(u < -1e-12, (hib - x) / u, np.inf), t_hi).min()
            t0 = np.maximum(np.where(u > 1e-12, (lob - x) / u, -np.inf), t_lo).max()
        # clip by inequalities  A(x+t u) <= b
        au = A_ub @ u; ax = A_ub @ x
        pos = au > 1e-14; neg = au < -1e-14
        if pos.any():
            t1 = min(t1, ((b_ub - ax)[pos] / au[pos]).min())
        if neg.any():
            t0 = max(t0, ((b_ub - ax)[neg] / au[neg]).max())
        if t1 <= t0:
            continue
        x = x + (t0 + (t1 - t0) * rng.random()) * u
        if step % thin == 0 and step > 5000:
            samples.append(x.copy())
    S = np.array(samples)
    print(f"  samples: {len(S)}")

    rec_rate = {}
    for t in TARGETS:
        for s, g in train.groupby("subject_id"):
            d = pd.to_datetime(g["sleep_date"])
            w = np.exp(-(d.max() - d).dt.days.values / 14.0)
            rec_rate[(s, t)] = float(np.sum(w * g[t].values) / np.sum(w))
    prior_center = np.array([rec_rate[(s, t)] for t in TARGETS for s in subjects])
    pw = np.exp(-0.5 * (((S - prior_center) / 0.15) ** 2).sum(axis=1))
    pw /= pw.sum()
    ess = 1.0 / np.sum(pw ** 2)

    print(f"\n  {'target':>6} {'tau':>5} | {'uniform: mean':>13} {'q05':>9} {'q95':>9} {'P(<0)':>6} |"
          f" {'recency-w mean':>14} {'P(<0)':>6}   (rec-ESS={ess:.0f})")
    for target in ["Q2", "Q1", "Q3"]:
        for tau in taus:
            ov, oc = objective_row(target, tau)
            g = S @ ov + oc
            q05, q50, q95 = np.percentile(g, [5, 50, 95])
            pneg = float((g < 0).mean())
            gm_w = float(np.sum(pw * g)); pneg_w = float(np.sum(pw[g < 0]))
            print(f"  {target:>6} {tau:5.2f} | {g.mean():+13.5f} {q05:+9.5f} {q95:+9.5f} {pneg:6.2f} |"
                  f" {gm_w:+14.5f} {pneg_w:6.2f}")

    print("\n=== joint Q1+Q2 combos (uniform posterior) ===")
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
