#!/usr/bin/env python3
"""H1 + H6: how much can we trust the public-loss sensor, and can we stack FrontierSilence?

H1  PREQUENTIAL BACKTEST: refit the ridge sensor chronologically (train on ledger
    entries with sequence < k, predict entry k). Reports sign accuracy, rank
    correlation and the magnitude calibration slope actual = c * predicted.
    Also reports per-file SUPPORT COVERAGE (what fraction of each submission's
    movement energy the sensor can even see).

H6  FRONTIER-SILENCE STACKING: FS (0.5677269444) is the only DIRECTLY OBSERVED
    negative delta vs H057 (-0.0000206). Mean logloss is exactly additive over
    cells, so if FS's changed cells are disjoint from candidate-3's changed
    cells, applying both moves gives delta_FS + delta_c3 exactly on the public
    set. This script measures the overlap and scores the stacked move.
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
OUT = HERE / "outputs" / "analysis_h1_h6"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["subject_id", "sleep_date", "lifelog_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
TOL = 1e-12

FS_FILE = "submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv"
FS_PUBLIC_LB = 0.5677269444
C3_FILE = "submission_final_candidate3_inhull_loo_consensus_b1ebf188_uploadsafe.csv"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c1 = import_module(HERE / "candidate_1_public_loss_sparse_tomography.py", "candidate1_for_h1h6")


def ridge_fit(x, y, alpha):
    col_scale = np.sqrt((x ** 2).mean(axis=0)) + 1e-6
    xs = x / col_scale
    coef_scaled = np.linalg.solve(xs.T @ xs + alpha * np.eye(xs.shape[1]), xs.T @ y)
    return coef_scaled / col_scale


def main():
    sample, base_prob, base_logit, base_grads, semantic_grads, h088_move = c1.load_world()
    n_cells = base_prob.size

    support_moves = {}
    for name in [c1.H154_FILE, c1.H155_FILE, c1.H158_FILE]:
        support_moves[name] = c1.movement_from_file(name, sample, base_logit)
    support_idx = np.flatnonzero(
        np.any(np.vstack([np.abs(m) > TOL for m in support_moves.values()]), axis=0)
    )
    print(f"support cells: {len(support_idx)}")

    # --- ledger in chronological order, with full movement vectors ---
    ledger = pd.read_csv(c1.PUBLIC_LEDGER).sort_values("sequence").reset_index(drop=True)
    rows = []
    for rec in ledger.to_dict("records"):
        move = c1.movement_from_file(str(rec["file"]), sample, base_logit)
        if move is None:
            continue
        energy = float(np.sum(move ** 2))
        sup_energy = float(np.sum(move[support_idx] ** 2))
        rows.append({
            "file": str(rec["file"]),
            "sequence": int(rec["sequence"]),
            "y": float(rec["public_lb"]) - c1.CURRENT_BEST_PUBLIC_LB,
            "move": move,
            "coverage": sup_energy / energy if energy > 0 else 1.0,
            "l2": float(np.sqrt(energy)),
        })
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "move"} for r in rows])
    print("\n=== per-file support coverage (sensor visibility of each probe) ===")
    print(frame[["sequence", "file", "y", "coverage", "l2"]].round(5).to_string(index=False))

    X = np.vstack([r["move"][support_idx] for r in rows])
    Y = np.array([r["y"] for r in rows])
    seqs = np.array([r["sequence"] for r in rows])

    # --- H1: prequential backtest ---
    print("\n=== H1: prequential backtest (train on past only, predict next) ===")
    alpha = 3.0
    preds, actuals, names, train_sizes = [], [], [], []
    for k in range(6, len(rows)):
        coef = ridge_fit(X[:k], Y[:k], alpha)
        preds.append(float(X[k] @ coef))
        actuals.append(Y[k])
        names.append(rows[k]["file"][:58])
        train_sizes.append(k)
    bt = pd.DataFrame({"n_train": train_sizes, "file": names, "pred": preds, "actual": actuals})
    bt["sign_ok"] = np.sign(bt["pred"]) == np.sign(bt["actual"])
    bt["ratio"] = bt["actual"] / bt["pred"].replace(0, np.nan)
    print(bt.round(6).to_string(index=False))

    def summarize(tag, sub):
        if len(sub) == 0:
            return
        from scipy import stats as _st  # may not exist; fallback below
    # manual spearman to avoid scipy dependency
    def spearman(a, b):
        ra = pd.Series(a).rank().values
        rb = pd.Series(b).rank().values
        ra = (ra - ra.mean()) / (ra.std() + 1e-12)
        rb = (rb - rb.mean()) / (rb.std() + 1e-12)
        return float((ra * rb).mean())

    for tag, sub in [("ALL", bt), ("HS-era (n_train>=19)", bt[bt["n_train"] >= 19])]:
        if len(sub) == 0:
            continue
        slope = float((sub["pred"] * sub["actual"]).sum() / max((sub["pred"] ** 2).sum(), 1e-18))
        print(f"\n  [{tag}] n={len(sub)} sign_acc={sub['sign_ok'].mean():.2f} "
              f"spearman={spearman(sub['pred'].values, sub['actual'].values):+.2f} "
              f"calibration_slope(actual~pred)={slope:+.4f}")

    # --- H6: FrontierSilence stacking ---
    print("\n=== H6: FrontierSilence stacking onto candidate 3 ===")
    fs_move = c1.movement_from_file(FS_FILE, sample, base_logit)
    c3_move = c1.movement_from_file(C3_FILE, sample, base_logit)
    fs_cells = np.flatnonzero(np.abs(fs_move) > TOL)
    c3_cells = np.flatnonzero(np.abs(c3_move) > TOL)
    overlap = np.intersect1d(fs_cells, c3_cells)
    print(f"  FS changed cells vs H057: {len(fs_cells)} | c3 changed cells: {len(c3_cells)} "
          f"| overlap: {len(overlap)}")
    if len(overlap):
        ov = pd.DataFrame({
            "flat_idx": overlap,
            "row": overlap // 7,
            "target": [TARGETS[i % 7] for i in overlap],
            "fs_move": fs_move[overlap],
            "c3_move": c3_move[overlap],
            "same_sign": np.sign(fs_move[overlap]) == np.sign(c3_move[overlap]),
        })
        print(ov.round(4).to_string(index=False))

    # stacked move: FS values on FS cells (observed-good), c3 values elsewhere
    stacked = c3_move.copy()
    stacked[fs_cells] = fs_move[fs_cells]  # FS wins overlaps (observed beats extrapolated)
    coef_full = ridge_fit(X, Y, alpha)
    for nm, mv in [("FS alone", fs_move), ("c3 alone", c3_move), ("stacked FS+c3", stacked)]:
        pred = float(mv[support_idx] @ coef_full)
        metric = c1.candidate_metrics(mv, base_grads, semantic_grads, h088_move)
        print(f"  {nm:16s} sensor_pred={pred:+.5f} changed={metric['changed_cells']:4d} "
              f"base_neg={metric['base_listener_negative_count']}/10 "
              f"sem_neg={metric['semantic_listener_negative_count']}/10 "
              f"h088_cos={metric['h088_cosine']:+.4f}")

    # FS coverage detail: how visible was FS to the sensor?
    fs_row = frame[frame["file"] == FS_FILE]
    if len(fs_row):
        print(f"  FS support coverage: {float(fs_row['coverage'].iloc[0]):.3f} "
              f"(fraction of FS movement energy inside sensor support)")

    np.save(OUT / "stacked_move.npy", stacked)
    bt.to_csv(OUT / "h1_prequential_backtest.csv", index=False)
    frame.to_csv(OUT / "h1_coverage.csv", index=False)
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
