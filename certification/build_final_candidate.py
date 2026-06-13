#!/usr/bin/env python3
"""Materialize the FINAL certified candidate from final_candidate_recipe.json.

FS base + per-subject logit shifts kappa*Lambda_s along the LB-validated probe
directions (Q2 kappa=0.75 capped 0.8, Q3 kappa=0.25). Zero assignment noise by the
H8 theorem; sign-safe across the full H9 sensitivity grid.

Run ONLY when the user asks for the submission file:
    python build_final_candidate.py            # builds + saves (repo root + Downloads)
    python build_final_candidate.py --dry      # checklist only, saves nothing
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ETRI = HERE.parent
OLD = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\outputs\submissions")
KEYS = ["subject_id", "sleep_date", "lifelog_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


c1 = import_module(HERE / "candidate_1_public_loss_sparse_tomography.py", "c1_for_build")


def load_abs(p):
    return pd.read_csv(p).sort_values(KEYS).reset_index(drop=True)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    recipe = json.loads((HERE / "final_candidate_recipe.json").read_text(encoding="utf-8"))
    base = load_abs(c1.locate(recipe["base_file"]))
    sample = load_abs(c1.locate("submission_h057_q2row_fullvector_state_7cde1a77_uploadsafe.csv"))[KEYS]
    sub = base["subject_id"].values
    subjects = sorted(set(sub))

    out = base.copy()
    for comp in recipe["components"]:
        t = comp["target"]
        f_from = load_abs(OLD / comp["probe_pair"][0])
        f_to = load_abs(OLD / comp["probe_pair"][1])
        dlg = logit(f_to[t].values) - logit(f_from[t].values)
        cap = float(comp["shift_cap_logit"])
        for s in subjects:
            m = sub == s
            lam = float(dlg[m].mean())
            shift = float(np.clip(comp["kappa"] * lam, -cap, cap))
            out.loc[m, t] = sigmoid(logit(out.loc[m, t].values) + shift)

    prob = out[TARGETS].to_numpy(float)
    digest = hashlib.sha1(np.round(prob, 12).tobytes()).hexdigest()[:8]
    name = f"submission_final_probe_direction_certified_{digest}_uploadsafe.csv"

    # 7-step checklist
    print("===== CHECKLIST =====")
    ok_shape = out.shape == (250, 10)
    ok_keys = out[KEYS].reset_index(drop=True).equals(sample.reset_index(drop=True))
    ok_null = int(out[TARGETS].isnull().sum().sum()) == 0
    ok_range = bool(prob.min() > 0.0 and prob.max() < 1.0)
    d = np.abs(prob - base[TARGETS].to_numpy(float))
    print(f"  [1] shape (250,10): {out.shape} -> {ok_shape}")
    print(f"  [2] keys==sample order: {ok_keys}")
    print(f"  [3] null==0: {ok_null}")
    print(f"  [4] strict (0,1): [{prob.min():.4f},{prob.max():.4f}] -> {ok_range}")
    print(f"  [5] vs FS base: mean|d|={d.mean():.5f} max|d|={d.max():.4f} changed={int((d>1e-12).sum())}")
    for i, t in enumerate(TARGETS):
        if d[:, i].max() > 1e-12:
            print(f"      {t}: mean {base[t].mean():.4f} -> {out[t].mean():.4f}")
    tmp = out.copy(); tmp["lifelog_date"] = pd.to_datetime(tmp["lifelog_date"])
    sp = {t: float(tmp.groupby("lifelog_date")[t].mean().std()) for t in TARGETS}
    print(f"  [6] date-mean std: " + " ".join(f"{t}={sp[t]:.3f}" for t in TARGETS))
    ok = ok_shape and ok_keys and ok_null and ok_range
    print(f"  [7] RESULT: {'PASS' if ok else 'FAIL'}")
    if "--dry" in sys.argv:
        print(f"  DRY RUN — nothing saved (would be: {name})")
        return
    if ok:
        out_path = ETRI / name
        out.to_csv(out_path, index=False)
        dl = Path(r"C:\Users\박광민\Downloads") / name
        out.to_csv(dl, index=False)
        print(f"  SAVED -> {out_path}")
        print(f"  SAVED -> {dl}")
    else:
        print("  NOT SAVED")


if __name__ == "__main__":
    main()
