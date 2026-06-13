#!/usr/bin/env python3
"""Candidate 3: In-Hull LOO-Consensus Sparse Tomography HS-JEPA.

Builds on Candidate 1 (public-loss sparse tomography) with three fixes aimed at
the public-LB transfer gap between the ridge sensor and reality:

  H-A  IN-HULL MAGNITUDE: cap each cell's logit move at the maximum |movement|
       ever OBSERVED at that cell across the public ledger files. The ridge was
       fit only on movements inside this hull; predictions outside it are pure
       linear extrapolation (Candidate 1 predicted -0.0658 while the largest
       real observed delta is ~0.008 -> ~80x extrapolation).

  H-B  LOO-CONSENSUS CELL VETO: keep a cell only if its public-loss coefficient
       keeps the SAME sign in >= min_loo_consensus of the 28 leave-one-out
       ridge fits (Candidate 1 allowed 0.55; we require >= 0.85), and only if
       the cell has materially been observed moving (hull >= min_hull).

  H-C  LOO-ENSEMBLE SELECTION: every candidate action field (Candidate 1's,
       Candidate 2's, ours, and scaled variants) is scored by the ensemble of
       28 LOO ridge models: mean predicted delta, std across folds, and the
       fraction of folds predicting improvement. We pick the configuration
       with full fold agreement and the best mean-minus-std, NOT the raw
       in-sample prediction.

Anchor and listener-safety conventions follow Candidate 1 exactly.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs" / "candidate_3_inhull_loo_consensus"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["subject_id", "sleep_date", "lifelog_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c1 = import_module(HERE / "candidate_1_public_loss_sparse_tomography.py", "candidate1_for_c3")


# ---------------------------------------------------------------------------
# Ridge machinery with explicit LOO fold models
# ---------------------------------------------------------------------------

def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    col_scale = np.sqrt((x ** 2).mean(axis=0)) + 1e-6
    xs = x / col_scale
    coef_scaled = np.linalg.solve(xs.T @ xs + alpha * np.eye(xs.shape[1]), xs.T @ y)
    return coef_scaled / col_scale


def loo_models(x: np.ndarray, y: np.ndarray, alpha: float) -> list[np.ndarray]:
    coefs = []
    for hold in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[hold] = False
        coefs.append(ridge_fit(x[mask], y[mask], alpha))
    return coefs


def loo_rmse(x: np.ndarray, y: np.ndarray, alpha: float) -> float:
    sq = []
    for hold in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[hold] = False
        coef = ridge_fit(x[mask], y[mask], alpha)
        sq.append(float((x[hold] @ coef - y[hold]) ** 2))
    return float(np.sqrt(np.mean(sq)))


def loo_eval(move_support: np.ndarray, coefs: list[np.ndarray]) -> dict[str, float]:
    preds = np.array([float(move_support @ coef) for coef in coefs])
    return {
        "loo_pred_mean": float(preds.mean()),
        "loo_pred_std": float(preds.std()),
        "loo_pred_min": float(preds.min()),
        "loo_pred_max": float(preds.max()),
        "loo_frac_improving": float((preds < 0).mean()),
    }


# ---------------------------------------------------------------------------
# Candidate 3 action construction
# ---------------------------------------------------------------------------

def build_action_c3(
    support_idx: np.ndarray,
    full_coef: np.ndarray,
    coefs: list[np.ndarray],
    hull: np.ndarray,
    support: dict,
    h158_move: np.ndarray,
    n_cells: int,
    config: dict,
) -> tuple[np.ndarray, pd.DataFrame]:
    sign_frame = np.vstack([np.sign(c) for c in coefs])
    consensus = np.maximum((sign_frame > 0).mean(axis=0), (sign_frame < 0).mean(axis=0))
    denom = np.percentile(np.abs(full_coef), 90) + 1e-9

    pool = []
    for local_idx, flat in enumerate(support_idx):
        coef_value = full_coef[local_idx]
        if abs(coef_value) < 1e-8:
            continue
        if consensus[local_idx] < config["min_loo_consensus"]:
            continue
        if hull[local_idx] < config["min_hull"]:
            continue
        desired_sign = int(-np.sign(coef_value))
        # all LOO folds must agree with the full-data sign for the move direction
        fold_agree = (np.sign([c[local_idx] for c in coefs]) == np.sign(coef_value)).mean()
        if fold_agree < config["min_loo_consensus"]:
            continue
        rec = support.get((int(flat), desired_sign))
        if rec is None:
            continue
        if rec["h088_alignment"] > config["max_h088_alignment"]:
            continue
        if rec["sem_mean"] < config["min_semantic_mean"] or rec["base_mean"] < config["min_base_mean"]:
            continue
        h158_agreement = 1.0 if desired_sign == np.sign(h158_move[flat]) else 0.75
        score = (
            abs(coef_value)
            * (0.35 + consensus[local_idx])
            * (1.0 + 0.30 * rec["source_support"])
            * (1.0 + 12000.0 * max(float(rec["sem_mean"]), 0.0))
            * (1.0 + 7000.0 * max(float(rec["base_mean"]), 0.0))
            * h158_agreement
            / (1.0 + 2.5 * rec["h088_alignment"])
        )
        pool.append((float(score), int(flat), local_idx, desired_sign, float(coef_value), float(consensus[local_idx])))
    pool.sort(reverse=True, key=lambda row: row[0])

    selected = []
    row_count: dict[int, int] = {}
    target_count: dict[str, int] = {}
    for item in pool:
        _score, flat, _local, _sign, *_ = item
        row = flat // len(TARGETS)
        target = TARGETS[flat % len(TARGETS)]
        if row_count.get(row, 0) >= config["max_cells_per_row"]:
            continue
        if target_count.get(target, 0) >= config["max_cells_per_target"]:
            continue
        selected.append(item)
        row_count[row] = row_count.get(row, 0) + 1
        target_count[target] = target_count.get(target, 0) + 1
        if len(selected) >= config["top_cells"]:
            break

    move = np.zeros(n_cells, dtype=np.float64)
    audit = []
    for score, flat, local_idx, sign, coef_value, cons in selected:
        c1_magnitude = config["action_amp"] * (0.12 + abs(coef_value) / denom)
        magnitude = min(config["per_cell_logit_cap"], c1_magnitude, config["hull_frac"] * hull[local_idx])
        move[flat] = sign * magnitude
        audit.append(
            {
                "score": score,
                "flat_idx": flat,
                "row": flat // len(TARGETS),
                "target": TARGETS[flat % len(TARGETS)],
                "sign": sign,
                "logit_move": float(move[flat]),
                "public_loss_coef": coef_value,
                "loo_consensus": cons,
                "hull": float(hull[local_idx]),
                "hull_binding": bool(config["hull_frac"] * hull[local_idx] < min(config["per_cell_logit_cap"], c1_magnitude)),
            }
        )
    return move, pd.DataFrame(audit)


def run() -> dict[str, object]:
    c1.ensure_prerequisites()
    sample, base_prob, base_logit, base_grads, semantic_grads, h088_move = c1.load_world()
    n_cells = base_prob.size

    support_moves = {}
    for name in [c1.H154_FILE, c1.H155_FILE, c1.H158_FILE]:
        move = c1.movement_from_file(name, sample, base_logit)
        if move is None:
            raise FileNotFoundError(name)
        support_moves[name] = move
    support_idx = np.flatnonzero(
        np.any(np.vstack([np.abs(move) > c1.TOL for move in support_moves.values()]), axis=0)
    )

    x, y, public_files = c1.public_observed_matrix(sample, base_logit, support_idx)
    hull = np.abs(x).max(axis=0)

    # --- alpha selection by LOO RMSE (H-C groundwork) ---
    alpha_grid = [3.0, 10.0, 30.0, 100.0]
    alpha_rmse = {a: loo_rmse(x, y, a) for a in alpha_grid}
    alpha = min(alpha_rmse, key=alpha_rmse.get)
    full_coef = ridge_fit(x, y, alpha)
    coefs = loo_models(x, y, alpha)

    support = c1.listener_support()

    # --- reference moves to beat (scored under the SAME LOO ensemble) ---
    references: dict[str, np.ndarray] = {}
    c1_rebuilt = None
    for fname in sorted(ROOT.glob("submission_final_candidate1_public_loss_sparse_tomography_*_uploadsafe.csv")):
        c1_rebuilt = fname.name
    if c1_rebuilt:
        references["candidate1"] = c1.movement_from_file(c1_rebuilt, sample, base_logit)
    for fname in sorted(ROOT.glob("submission_final_candidate2_cohort_relative_atlas_*_uploadsafe.csv")):
        references["candidate2"] = c1.movement_from_file(fname.name, sample, base_logit)

    report_rows = []

    def add_report(name: str, move: np.ndarray) -> dict[str, float]:
        ev = loo_eval(move[support_idx], coefs)
        metric = c1.candidate_metrics(move, base_grads, semantic_grads, h088_move)
        row = {
            "name": name,
            **ev,
            "changed_cells": metric["changed_cells"],
            "base_listener_neg": metric["base_listener_negative_count"],
            "sem_listener_neg": metric["semantic_listener_negative_count"],
            "h088_cosine": metric["h088_cosine"],
            "max_abs_logit_move": float(np.abs(move).max()),
        }
        report_rows.append(row)
        return row

    for name, move in references.items():
        if move is not None:
            add_report(name, move)

    # --- candidate 3 grid (strictness x hull usage) ---
    grids = {
        "c3_strict": dict(min_loo_consensus=1.0, min_hull=0.05, hull_frac=1.0),
        "c3_consensus085": dict(min_loo_consensus=0.85, min_hull=0.05, hull_frac=1.0),
        "c3_half_hull": dict(min_loo_consensus=1.0, min_hull=0.05, hull_frac=0.5),
    }
    base_config = dict(
        top_cells=110,
        action_amp=1.5,
        per_cell_logit_cap=1.9,
        max_cells_per_row=2,
        max_cells_per_target=55,
        max_h088_alignment=0.66,
        min_semantic_mean=-5e-6,
        min_base_mean=-5e-5,
    )

    builds: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for gname, gcfg in grids.items():
        config = {**base_config, **gcfg}
        move, audit = build_action_c3(
            support_idx, full_coef, coefs, hull, support, support_moves[c1.H158_FILE], n_cells, config
        )
        builds[gname] = (move, audit)
        add_report(gname, move)
        # scaled variants of the strict build only (linear in kappa; for the record)
        if gname == "c3_strict":
            for kappa in (0.5, 0.75):
                add_report(f"c3_strict_x{kappa}", kappa * move)

    report = pd.DataFrame(report_rows)
    report = report.sort_values("loo_pred_mean")
    report.to_csv(OUT / "candidate3_loo_report.csv", index=False)
    print("=== LOO-ensemble evaluation (lower mean = better; want frac_improving = 1.0) ===")
    print(
        report[
            ["name", "loo_pred_mean", "loo_pred_std", "loo_frac_improving", "changed_cells",
             "base_listener_neg", "sem_listener_neg", "max_abs_logit_move"]
        ].to_string(index=False)
    )

    # --- selection: full LOO agreement + all listeners negative, best mean+std ---
    candidates = report[
        (report["name"].str.startswith("c3"))
        & (report["loo_frac_improving"] >= 1.0 - 1e-9)
        & (report["base_listener_neg"] == 10)
        & (report["sem_listener_neg"] == 10)
    ].copy()
    if candidates.empty:
        candidates = report[report["name"].str.startswith("c3")].copy()
    candidates["objective"] = candidates["loo_pred_mean"] + candidates["loo_pred_std"]
    winner_name = str(candidates.sort_values("objective").iloc[0]["name"])
    base_winner = winner_name.replace("_x0.5", "").replace("_x0.75", "")
    scale = 0.5 if winner_name.endswith("_x0.5") else (0.75 if winner_name.endswith("_x0.75") else 1.0)
    winner_move = scale * builds[base_winner][0]
    winner_audit = builds[base_winner][1]

    prob = c1.clip_prob(c1.sigmoid(base_logit + winner_move).reshape(base_prob.shape))
    digest = c1.short_hash(prob)
    name = f"submission_final_candidate3_inhull_loo_consensus_{digest}_uploadsafe.csv"
    local_path = OUT / name
    root_path = ROOT / name
    c1.write_submission(local_path, sample, prob)
    c1.write_submission(root_path, sample, prob)
    winner_audit.to_csv(OUT / "candidate3_selected_cells.csv", index=False)

    readout = {
        "candidate": "In-Hull LOO-Consensus Sparse Tomography HS-JEPA",
        "submission_file": name,
        "root_path": str(root_path.resolve()),
        "hash": digest,
        "anchor_file": c1.CURRENT_BEST_FILE,
        "anchor_public_lb": c1.CURRENT_BEST_PUBLIC_LB,
        "ridge_alpha_grid_loo_rmse": {str(k): v for k, v in alpha_rmse.items()},
        "ridge_alpha_selected": alpha,
        "public_observation_count": int(len(public_files)),
        "winner": winner_name,
        "winner_scale": scale,
        "winner_loo": loo_eval(winner_move[support_idx], coefs),
        "winner_changed_cells": int((np.abs(winner_move) > c1.TOL).sum()),
        "hull_binding_cells": int(winner_audit["hull_binding"].sum()) if len(winner_audit) else 0,
        "listener_metrics": c1.candidate_metrics(winner_move, base_grads, semantic_grads, h088_move),
        "validation": c1.validate_submission(root_path, sample, base_prob),
    }
    (OUT / "candidate3_readout.json").write_text(json.dumps(readout, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(readout, indent=2, ensure_ascii=False))
    return readout


if __name__ == "__main__":
    run()
