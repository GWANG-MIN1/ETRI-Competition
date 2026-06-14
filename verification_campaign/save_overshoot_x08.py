#!/usr/bin/env python3
"""Build & SAVE the overshoot x0.8 submission (user requested the CSV).

This is the campaign's single best bet to lower public LB below the 0.5619 anchor:
FS base + 0.8 x (per-subject Q2/Q3 drift-overshoot field). Three independent methods agree
sigma~0.75-0.85 beats sigma=1.0 (the 0.5619 anchor) on expected public gain by ~0.0002-0.0004
and it is also private-safe (collinear scaling of the validated drift axis).
Saves an upload-safe 250x10 CSV and re-verifies upload-safety + polytope post-mean.
"""
from __future__ import annotations
from pathlib import Path
import sys
import hashlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polytope_eval as PE  # noqa: E402

KEYS = PE.KEYS
TARGETS = PE.TARGETS
EPS = 1e-6
OUT = PE.ETRI / "submission_overshoot_x0p8_330ef1a1_uploadsafe.csv"


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    PE.init()
    vix = PE._S["vix"]; subs = PE._S["subs"]; d_over = PE._S["d_over"]
    d = d_over * 0.8

    fs = pd.read_csv(PE.ETRI / PE.FS_FILE).sort_values(KEYS).reset_index(drop=True)
    sample = pd.read_csv(PE.ETRI / "data" / "ch2026_submission_sample.csv").sort_values(KEYS).reset_index(drop=True)
    fsub = fs["subject_id"].values
    base = fs[TARGETS].to_numpy(float)

    out = fs[KEYS].copy()
    for t in TARGETS:
        z = logit(fs[t].values).copy()
        for s in subs:
            z[fsub == s] += d[vix[(s, t)]]
        out[t] = sigmoid(z)

    # restore the sample's exact column order & row order
    out = out[sample.columns.tolist()] if set(sample.columns) == set(out.columns) else out
    out = out.set_index(KEYS).loc[sample.set_index(KEYS).index].reset_index()

    prob = out[TARGETS].to_numpy(float)
    digest = hashlib.sha1(np.round(prob, 12).tobytes()).hexdigest()[:8]
    dd = np.abs(prob - base)

    # upload-safety checklist
    checks = {
        "shape (250,10)": out.shape == (250, 10),
        "keys==sample": out[KEYS].reset_index(drop=True).equals(sample[KEYS].reset_index(drop=True)),
        "null==0": int(out[TARGETS].isnull().sum().sum()) == 0,
        "strictly in (0,1)": bool(prob.min() > 0 and prob.max() < 1),
        "only Q2/Q3 changed": [t for i, t in enumerate(TARGETS) if dd[:, i].max() > 1e-12] == ["Q2", "Q3"],
    }
    print("Upload-safety checklist:")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    assert all(checks.values()), "upload-safety FAILED — not saving"

    out.to_csv(OUT, index=False)
    print(f"\nSAVED: {OUT}")
    print(f"  hash {digest}  (expected 330ef1a1)")
    print(f"  rows {len(out)}  cols {list(out.columns)}")
    print(f"  Q2 mean {fs['Q2'].mean():.4f}->{out['Q2'].mean():.4f}  Q3 {fs['Q3'].mean():.4f}->{out['Q3'].mean():.4f}")
    print(f"  vs FS: mean|delta| {dd.mean():.5f}  changed cells {int((dd>1e-12).sum())}")

    # re-verify the expected public gain via the authoritative arbiter
    r = PE.eval_delta(d, label="overshoot x0.8 (saved)", settings18=True, verbose=False)
    print(f"\n  polytope: post-mean {r['post_mean']:+.5f}  worst-loose {r['worst_loose']:+.5f}  "
          f"fav18 {r['fav18']:.2f}  P(<0) {r['p_improve']:.2f}")
    anch = PE.eval_delta(d_over, verbose=False)
    print(f"  anchor x1.0 (=0.5619): post-mean {anch['post_mean']:+.5f}  worst-loose {anch['worst_loose']:+.5f}")
    print(f"  => x0.8 expected better than the anchor by {r['post_mean']-anch['post_mean']:+.5f} (post-mean).")
    print(f"  predicted public LB ~ 0.5619 + ({r['post_mean']-anch['post_mean']:+.5f}) ~ "
          f"{0.5619100863 + (r['post_mean']-anch['post_mean']):.5f} (model estimate).")


if __name__ == "__main__":
    main()
