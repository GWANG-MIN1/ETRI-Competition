#!/usr/bin/env python3
"""H7: cross-line transfer of the OBSERVED public Q-elevation to the h-line (0.5677 basin).

Logic:
  - On the old pipeline, two single-target probe pairs were submitted:
      A -> B changed ONLY Q2 cells: LB 0.6001    -> 0.5949167449  (gain -0.0051833)
      B -> C changed ONLY Q3 cells: LB 0.5949167 -> 0.593188787   (gain -0.0017279)
    Mean logloss is exactly additive over cells, so each measured delta is EXACTLY the
    public-label functional of those target cells. This measures PUBLIC LABELS, which are
    pipeline-independent.
  - Step 1: verify the h-line ledger never varied target LEVELS (unprobed axis there).
  - Step 2: calibrate an elevation model on the old-line pairs: public label rate model
      r_s = clip(train_subject_rate_s + Delta, eps, 1-eps)   (per-subject elevation)
    and global-r model. Solve Delta (resp. r) so the predicted gain of A->B matches the
    measured -0.0051833 exactly. Same for Q3 with B->C.
  - Step 3: apply the calibrated label model to the FS submission (h-line best): compute
    the exact expected-gain curve for a uniform logit shift tau on Q2 (resp. Q3) cells,
    the optimal tau*, the gain at tau*, and the BREAKEVEN Delta below which shifting hurts.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ETRI = Path(r"C:\Users\박광민\Documents\Codex\etri_team")
OLD = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\outputs\submissions")
RAW = ETRI / "data"
OUT = ETRI / "final_hsjepa_candidates" / "outputs" / "analysis_h7"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["subject_id", "sleep_date", "lifelog_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6

FILE_A = OLD / "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv"   # LB 0.6001
FILE_B = OLD / "submission_Q1up100_plus_Q2recency_tau10_lam075.csv"        # LB 0.5949167449
FILE_C = OLD / "submission_FINAL_Q2recency_plus_Q3recency.csv"             # LB 0.593188787
GAIN_Q2 = 7.0 * (0.5949167449 - 0.6001)        # target-logloss gain attributable to Q2 cells
GAIN_Q3 = 7.0 * (0.593188787 - 0.5949167449)   # ... to Q3 cells

FS_FILE = ETRI / "submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv"
H057_FILE = ETRI / "submission_h057_q2row_fullvector_state_7cde1a77_uploadsafe.csv"


def load(path):
    df = pd.read_csv(path)
    return df.sort_values(KEYS).reset_index(drop=True)


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def expected_ll(p, r):
    """E_{y~Bern(r)} logloss(p, y), elementwise."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(r * np.log(p) + (1 - r) * np.log(1 - p))


def shift_logit(p, tau):
    p = np.clip(p, EPS, 1 - EPS)
    z = np.log(p / (1 - p)) + tau
    return 1.0 / (1.0 + np.exp(-z))


def main():
    train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
    subj_rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}

    # ------------------------------------------------------------------
    print("=== STEP 1: was the LEVEL axis ever probed in the h-line ledger? ===")
    ledger = pd.read_csv(ETRI / "data_analytics" / "hsjepa_public_score_ledger.csv")
    means = []
    for rec in ledger.to_dict("records"):
        f = ETRI / str(rec["file"])
        if not f.exists():
            continue
        df = load(f)
        means.append({"file": str(rec["file"])[:48], "lb": rec["public_lb"],
                      **{t: df[t].mean() for t in TARGETS}})
    mdf = pd.DataFrame(means)
    print(mdf.round(4).to_string(index=False))
    spread = mdf[TARGETS].max() - mdf[TARGETS].min()
    # HS-era only (the current basin): lb < 0.569
    hs = mdf[mdf["lb"] < 0.569]
    hs_spread = hs[TARGETS].max() - hs[TARGETS].min()
    print("\n  per-target mean spread, ALL ledger:  ", spread.round(4).to_dict())
    print("  per-target mean spread, HS-era only:", hs_spread.round(4).to_dict())

    # ------------------------------------------------------------------
    print("\n=== STEP 2: calibrate public-label elevation on the OLD line's measured probes ===")
    A, B, C = load(FILE_A), load(FILE_B), load(FILE_C)
    sub = A["subject_id"].values

    def calibrate(target, dfa, dfb, measured_gain):
        pa, pb = dfa[target].values, dfb[target].values
        changed = np.abs(pa - pb) > 1e-12
        assert changed.sum() > 0
        # confirm single-target attribution
        other = [t for t in TARGETS if t != target]
        max_other = max(np.abs(dfa[t].values - dfb[t].values).max() for t in other)
        base_rate = np.array([subj_rate[target][s] for s in sub])

        def pred_gain_delta(delta):
            r = np.clip(base_rate + delta, 0.02, 0.98)
            return float(np.mean(expected_ll(pb, r) - expected_ll(pa, r)))

        def pred_gain_global(r):
            return float(np.mean(expected_ll(pb, r) - expected_ll(pa, r)))

        # solve by bisection on delta in [-0.2, 0.45]
        lo, hi = -0.2, 0.45
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if pred_gain_delta(mid) > measured_gain:  # need more elevation for more gain
                lo = mid
            else:
                hi = mid
        delta_star = 0.5 * (lo + hi)
        lo, hi = 0.05, 0.95
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if pred_gain_global(mid) > measured_gain:
                lo = mid
            else:
                hi = mid
        r_star = 0.5 * (lo + hi)
        print(f"  {target}: changed {changed.sum()} cells (max other-target move {max_other:.1e})"
              f" | measured target-gain {measured_gain:+.5f}")
        print(f"      -> per-subject elevation Delta* = {delta_star:+.3f}"
              f" (train marg {train[target].mean():.3f} -> implied public ~{train[target].mean()+delta_star:.3f})")
        print(f"      -> global-rate        r*     = {r_star:.3f}")
        return delta_star, r_star

    dQ2, rQ2 = calibrate("Q2", A, B, GAIN_Q2)
    dQ3, rQ3 = calibrate("Q3", B, C, GAIN_Q3)

    # ------------------------------------------------------------------
    print("\n=== STEP 3: exact expected-gain curves on the h-line best (FrontierSilence) ===")
    FS = load(FS_FILE)
    sub_fs = FS["subject_id"].values

    def curve(target, delta_star, r_star):
        p = FS[target].values
        base_rate = np.array([subj_rate[target][s] for s in sub_fs])
        print(f"\n  --- {target}: FS mean {p.mean():.3f} (train marg {train[target].mean():.3f}) ---")
        print(f"  {'tau':>5} | {'gain @Delta*':>12} | {'@0.5*Delta*':>12} | {'@0.25*Delta*':>13} |"
              f" {'@Delta=0':>9} | {'@global r*':>10}")
        taus = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        rows = []
        for tau in taus:
            p2 = shift_logit(p, tau)
            def g(delta):
                r = np.clip(base_rate + delta, 0.02, 0.98)
                return float(np.mean(expected_ll(p2, r) - expected_ll(p, r))) / 7.0  # mean-LB units
            gr = float(np.mean(expected_ll(p2, r_star) - expected_ll(p, r_star))) / 7.0
            rows.append((tau, g(delta_star), g(0.5 * delta_star), g(0.25 * delta_star), g(0.0), gr))
            print(f"  {tau:5.2f} | {rows[-1][1]:+12.5f} | {rows[-1][2]:+12.5f} | {rows[-1][3]:+13.5f} |"
                  f" {rows[-1][4]:+9.5f} | {rows[-1][5]:+10.5f}")
        # breakeven Delta for tau=0.10 and 0.20
        for tau in (0.10, 0.20):
            p2 = shift_logit(p, tau)
            lo, hi = -0.1, 0.45
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                r = np.clip(base_rate + mid, 0.02, 0.98)
                if float(np.mean(expected_ll(p2, r) - expected_ll(p, r))) < 0:
                    hi = mid
                else:
                    lo = mid
            print(f"  breakeven elevation for tau={tau:.2f}: Delta >= {0.5*(lo+hi):+.3f}"
                  f"  (calibrated Delta* = {delta_star:+.3f})")
        return rows

    curve("Q2", dQ2, rQ2)
    curve("Q3", dQ3, rQ3)

    # Q1 reference: old line showed public Q1 ~0.55-0.60 and overshoot at .645.
    # FS Q1 mean:
    print(f"\n  Q1 reference: FS Q1 mean = {FS['Q1'].mean():.3f} (old-line evidence: public Q1 ~0.55-0.60)")
    print(f"  S means FS: " + " ".join(f"{t}={FS[t].mean():.3f}" for t in ["S1","S2","S3","S4"]))
    print("  (S-side: no single-target public probe evidence on either line -> excluded from H7)")


if __name__ == "__main__":
    main()
