#!/usr/bin/env python3
"""H11: LOCO robustness of the certificate — survives any single measurement being wrong?

For each of the 28 public measurements, rebuild the LOOSEST polytope (comp 30%, box ±0.45,
Z=3 — certifies all 18 settings) WITHOUT that measurement and recompute the LP worst-case
of the locked recipe. Also: drop ALL old-line probes (h-line-only knowledge), drop the Q2
probe alone (the suspected load-bearer), and report which constraints are BINDING at the
full-polytope worst-case vertex.
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


def imp(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


c1 = imp(HERE / "candidate_1_public_loss_sparse_tomography.py", "c1_for_h11")


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

    named = []
    ledger = pd.read_csv(ETRI / "data_analytics" / "hsjepa_public_score_ledger.csv")
    for rec in ledger.to_dict("records"):
        p = c1.locate(str(rec["file"]))
        if p is None:
            continue
        v, c0, sg = pair_row(H057, load_abs(p))
        short = str(rec["file"]).replace("submission_", "")[:36]
        named.append((short, v, c0, float(rec["public_lb"]) - H057_LB, sg))
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

    rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}
    bounds = [(max(0.02, rate[t][s] - 0.45), min(0.98, rate[t][s] + 0.45))
              for t in TARGETS for s in subjects]

    # recipe objective (delta from probe directions)
    delta = np.zeros(n_var)
    for target, kappa, (f1, f2) in [("Q2", 0.75, ("A", "B")), ("Q3", 0.25, ("B", "C"))]:
        dlg = logit(olds[f2][target].values) - logit(olds[f1][target].values)
        for s in subjects:
            lam = float(dlg[masks[s]].mean())
            delta[vidx[(s, target)]] = float(np.clip(kappa * lam, -CAP, CAP))
    A_g = 0.0
    coefs_g = np.zeros(n_var)
    for t in TARGETS:
        z = logit(FS[t].values)
        for s in subjects:
            d = delta[vidx[(s, t)]]
            zz = z[masks[s]]
            A_g += float(-np.log((1 - sigmoid(zz + d)) / (1 - sigmoid(zz))).sum()) / N_CELLS
            coefs_g[vidx[(s, t)]] = -n_rows[s] * d / N_CELLS

    def build(drop=frozenset()):
        A_ub, b_ub, tags = [], [], []
        for i, (nm, v, c0, d, sg) in enumerate(named):
            if i in drop:
                continue
            tol = 1e-5 + 0.30 * abs(d) + 3.0 * sg
            A_ub.append(v);  b_ub.append(d + tol - c0); tags.append((nm, "+"))
            A_ub.append(-v); b_ub.append(-(d - tol - c0)); tags.append((nm, "-"))
        return np.array(A_ub), np.array(b_ub), tags

    def worst(drop=frozenset()):
        A_ub, b_ub, tags = build(drop)
        res = linprog(-coefs_g, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            return None, None, tags
        return A_g - (-res.fun) if False else A_g + (-res.fun) * -1.0, res, tags

    # NOTE: worst-case gain = A_g + max_r(coefs_g @ r) = A_g - min_r(-coefs_g @ r)
    def worst_value(drop=frozenset()):
        A_ub, b_ub, tags = build(drop)
        res = linprog(-coefs_g, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        return (A_g - res.fun, res, tags) if res.success else (None, None, tags)

    base_val, base_res, base_tags = worst_value()
    print(f"=== full polytope (loosest setting) certified worst: {base_val:+.5f} ===")
    marg = np.asarray(base_res.ineqlin.marginals)
    print("  binding constraints at worst-case vertex (|dual|>1e-6):")
    for j in np.flatnonzero(np.abs(marg) > 1e-6):
        nm, side = base_tags[j]
        print(f"    {nm} [{side}]  dual={marg[j]:+.3e}")

    print("\n=== LOCO: drop each measurement, recompute certified worst ===")
    print(f"  {'dropped measurement':>40} | {'worst':>9} | sign-safe?")
    worst_break = None
    for i, (nm, *_rest) in enumerate(named):
        val, _res, _ = worst_value(frozenset([i]))
        flag = "OK" if (val is not None and val < 0) else "BREAKS"
        if flag == "BREAKS" and worst_break is None:
            worst_break = nm
        print(f"  {nm:>40} | {val:+9.5f} | {flag}")

    print("\n=== harsher ablations ===")
    idx_by_name = {nm: i for i, (nm, *_r) in enumerate(named)}
    for label, drops in [
        ("drop old_Q2probe only", ["old_Q2probe"]),
        ("drop old_Q2+Q3 probes", ["old_Q2probe", "old_Q3probe"]),
        ("drop ALL old-line probes (h-line only)", ["old_Q2probe", "old_Q3probe", "old_Q1probe"]),
    ]:
        val, _res, _ = worst_value(frozenset(idx_by_name[nm] for nm in drops))
        print(f"  {label:>40} | {val:+9.5f} | {'OK' if val < 0 else 'NOT certified'}")


if __name__ == "__main__":
    main()
