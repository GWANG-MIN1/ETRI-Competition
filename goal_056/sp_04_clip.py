"""Strategy-doc Step 4 last stone: per-target probability CLIPPING re-exploration (the
doc's safest 'pure calibration' lever) + does the best (FS overshoot x0.8) already absorb
the Q2/Q3 level so calibration has no room above it?

Honest CV (test_faithful + interleaved), train-fold derived clip applied to held-out base.
Also: residual room above the best analog (fs_proxy 0.8) on every target.
"""
import sys
from pathlib import Path
import numpy as np

KIT_DIR = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(HSJEPA_SRC)); sys.path.insert(0, str(KIT_DIR))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa

EPS = 1e-6
TARGETS = list(H.TARGETS)
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173]


def clip_delta(m, proxy):
    clips = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    res = {t: {c: [] for c in clips} for t in TARGETS}
    for sd in SEEDS:
        if proxy == "test_faithful":
            held = H.test_faithful_mask(m, sd); splits = [(np.flatnonzero(~held), np.flatnonzero(held))]
        else:
            splits = H.interleaved_folds(m, 5, sd)
        for t in TARGETS:
            y = m[f"y_{t}"].values
            for tr, va in splits:
                p = m[f"p_{t}"].values[va]
                base_ll = H.bll(y[va], p)
                for c in clips:
                    res[t][c].append(H.bll(y[va], np.clip(p, c, 1 - c)) - base_ll)
    print(f"\n=== per-target clipping delta [{proxy}] (vs base, <0 better) ===")
    print("  tgt | " + " | ".join([f"c={c}" for c in clips]))
    for t in TARGETS:
        cells = [f"{np.mean(res[t][c]):+.5f}" for c in clips]
        best_c = min(clips, key=lambda c: np.mean(res[t][c]))
        print(f"  {t}  | " + " | ".join(cells) + f"  best c={best_c}")


def room_above_best(m, proxy):
    """Honest: does anything beat fs_proxy(0.8) (best analog)? We compare base vs anchor and
    show anchor's CV delta per target (negative=anchor better on this CV)."""
    anchor = K.fs_proxy_arr(m, overshoot=0.8)
    print(f"\n=== anchor (FS overshoot x0.8) vs base [{proxy}] per-target CV delta ===")
    res = {t: [] for t in TARGETS}
    for sd in SEEDS:
        if proxy == "test_faithful":
            held = H.test_faithful_mask(m, sd); splits = [(None, np.flatnonzero(held))]
        else:
            splits = H.interleaved_folds(m, 5, sd)
        for t in TARGETS:
            y = m[f"y_{t}"].values
            ti = TARGETS.index(t)
            for _, va in splits:
                res[t].append(H.bll(y[va], anchor[va, ti]) - H.bll(y[va], m[f"p_{t}"].values[va]))
    print("  tgt | anchor-delta (note: Q-up overshoot is an LB-probe, CV penalizes it)")
    for t in TARGETS:
        print(f"  {t}  | {np.mean(res[t]):+.5f}")


if __name__ == "__main__":
    m = K.base()
    for proxy in ("test_faithful", "interleaved"):
        clip_delta(m, proxy)
    for proxy in ("test_faithful", "interleaved"):
        room_above_best(m, proxy)
    print("\nDONE")
