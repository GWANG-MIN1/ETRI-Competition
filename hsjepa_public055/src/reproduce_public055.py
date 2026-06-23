#!/usr/bin/env python3
"""Reproduce the first public-0.55 HS-JEPA Listener submission.

This script is intentionally self-contained and conservative.

It does not depend on the original large experiment tree.  It replays the
confirmed two-regime row-target assignment action:

    conflict_rescue_base_public_0p5612880941.csv
      + two_regime_assignment_union_k360_selected_cells.csv
      -> submission_public055_reproduced.csv

The reproduced file should match the expected public-0.55 submission up to
floating point precision and have stable target hash `dc3a5f9d53`.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEYS = ["subject_id", "sleep_date", "lifelog_date"]
EPS = 1e-5
EXPECTED_STABLE_HASH = "dc3a5f9d53"
EXPECTED_PUBLIC_LB = 0.5595724276


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def logit(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype="float64"), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-values))


def stable_target_hash(frame: pd.DataFrame) -> str:
    arr = np.round(frame[TARGETS].to_numpy(dtype="float64"), 10)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:10]


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(frame.columns) != list(sample.columns):
        raise ValueError("submission columns do not match sample submission")
    if len(frame) != len(sample):
        raise ValueError(f"row count mismatch: {len(frame)} != {len(sample)}")
    if not frame[KEYS].astype(str).equals(sample[KEYS].astype(str)):
        raise ValueError("submission key columns do not match sample submission")
    values = frame[TARGETS].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("submission contains non-finite probabilities")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("submission probabilities are outside [0, 1]")


def apply_selected_cells(base: pd.DataFrame, selected_cells: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    required = {"row_idx", "target", "candidate_logit_step"}
    missing = sorted(required - set(selected_cells.columns))
    if missing:
        raise ValueError(f"selected cell file missing required columns: {missing}")

    for rec in selected_cells.itertuples(index=False):
        row_idx = int(rec.row_idx)
        target = str(rec.target)
        if target not in TARGETS:
            raise ValueError(f"unknown target in selected cells: {target}")
        before = float(out.loc[row_idx, target])
        step = float(rec.candidate_logit_step)
        after = float(np.clip(sigmoid(logit([before])[0] + step), EPS, 1.0 - EPS))
        out.loc[row_idx, target] = after
    return out


def summarize_action(base: pd.DataFrame, reproduced: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["S1", "S2", "S3", "S4"]:
        mask = selected["target"].eq(target)
        changed = int(mask.sum())
        abs_delta = np.abs(
            reproduced.loc[:, target].to_numpy(dtype="float64")
            - base.loc[:, target].to_numpy(dtype="float64")
        )
        rows.append(
            {
                "target": target,
                "selected_cells": changed,
                "mean_abs_probability_delta": float(abs_delta[abs_delta > 0].mean())
                if np.any(abs_delta > 0)
                else 0.0,
                "max_abs_probability_delta": float(abs_delta.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=package_root() / "outputs" / "submission_public055_reproduced.csv",
        help="Output submission path.",
    )
    parser.add_argument(
        "--skip-expected-check",
        action="store_true",
        help="Only generate output; do not compare against packaged expected submission.",
    )
    args = parser.parse_args()

    root = package_root()
    sample_path = root / "data" / "ch2026_submission_sample.csv"
    base_path = root / "data" / "conflict_rescue_base_public_0p5612880941.csv"
    selected_path = root / "data" / "two_regime_assignment_union_k360_selected_cells.csv"
    expected_path = root / "outputs" / "submission_public055_reproduced_expected.csv"

    sample = pd.read_csv(sample_path)
    base = pd.read_csv(base_path)
    selected = pd.read_csv(selected_path)

    validate_submission(base, sample)
    if len(selected) != 360:
        raise ValueError(f"expected 360 selected cells, got {len(selected)}")

    reproduced = apply_selected_cells(base, selected)
    validate_submission(reproduced, sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    reproduced.to_csv(args.out, index=False)

    stable_hash = stable_target_hash(reproduced)
    if stable_hash != EXPECTED_STABLE_HASH:
        raise ValueError(f"stable hash mismatch: {stable_hash} != {EXPECTED_STABLE_HASH}")

    max_abs_diff = None
    if not args.skip_expected_check:
        expected = pd.read_csv(expected_path)
        validate_submission(expected, sample)
        max_abs_diff = float(
            np.max(
                np.abs(
                    reproduced[TARGETS].to_numpy(dtype="float64")
                    - expected[TARGETS].to_numpy(dtype="float64")
                )
            )
        )
        if max_abs_diff > 1e-12:
            raise ValueError(f"expected submission mismatch: max_abs_diff={max_abs_diff}")

    summary = summarize_action(base, reproduced, selected)
    print("Reproduced public-0.55 HS-JEPA Listener submission")
    print(f"- output: {args.out}")
    print(f"- expected public LB from submitted file: {EXPECTED_PUBLIC_LB:.10f}")
    print(f"- stable target hash: {stable_hash}")
    if max_abs_diff is not None:
        print(f"- max abs diff vs packaged expected submission: {max_abs_diff:.3e}")
    print("- selected S cells by target:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

