"""HS-JEPA exploration — Step 0: verify foundational facts.

Checks:
  1. train label type (binary vs continuous), targets, subjects
  2. test (submission sample) subjects vs train subjects
  3. current best submission per-target means
  4. what kbs56774.csv is relative to current best
  5. interleaving structure (test dates vs train dates per subject)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "raw"
SUBS = ROOT / "outputs" / "submissions"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

train = pd.read_csv(RAW / "ch2026_metrics_train.csv")
sample = pd.read_csv(RAW / "ch2026_submission_sample.csv")
best = pd.read_csv(SUBS / "submission_FINAL_Q2recency_plus_Q3recency.csv")
kbs = pd.read_csv(r"C:\Users\박광민\Downloads\kbs56774.csv")

print("=" * 70)
print("1. TRAIN LABELS")
print("  shape:", train.shape, "| cols:", list(train.columns))
print("  n subjects:", train["subject_id"].nunique(), "| subjects:", sorted(train["subject_id"].unique())[:12])
for t in TARGETS:
    vals = train[t].dropna()
    uniq = vals.unique()
    is_binary = set(np.unique(np.round(uniq, 6))) <= {0.0, 1.0}
    print(f"  {t}: n={len(vals)} mean={vals.mean():.4f} binary={is_binary} n_unique={len(uniq)} min={vals.min():.3f} max={vals.max():.3f}")

print("=" * 70)
print("2. TEST / SUBMISSION SAMPLE")
print("  shape:", sample.shape, "| cols:", list(sample.columns))
print("  n subjects:", sample["subject_id"].nunique(), "| subjects:", sorted(sample["subject_id"].unique())[:12])
tr_subj = set(train["subject_id"].unique())
te_subj = set(sample["subject_id"].unique())
print("  test subjects ⊆ train subjects?", te_subj <= tr_subj)
print("  test∩train:", len(te_subj & tr_subj), "| test-only:", sorted(te_subj - tr_subj), "| train-only:", sorted(tr_subj - te_subj))

print("=" * 70)
print("3. INTERLEAVING (test dates vs train dates, per subject)")
train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])
sample_dates = best.copy()
sample_dates["lifelog_date"] = pd.to_datetime(sample_dates["lifelog_date"])
n_within, n_future, n_past = 0, 0, 0
for s in sorted(te_subj):
    tr_d = train.loc[train.subject_id == s, "lifelog_date"]
    te_d = sample_dates.loc[sample_dates.subject_id == s, "lifelog_date"]
    if len(tr_d) == 0:
        continue
    lo, hi = tr_d.min(), tr_d.max()
    for d in te_d:
        if d < lo:
            n_past += 1
        elif d > hi:
            n_future += 1
        else:
            n_within += 1
tot = n_within + n_future + n_past
print(f"  test rows: within train span={n_within} ({n_within/tot:.1%}), future={n_future} ({n_future/tot:.1%}), past={n_past} ({n_past/tot:.1%})")

print("=" * 70)
print("4. CURRENT BEST per-target means (LB 0.5932)")
print("  ", {t: round(best[t].mean(), 4) for t in TARGETS})
print("  TRAIN marginal means")
print("  ", {t: round(train[t].mean(), 4) for t in TARGETS})

print("=" * 70)
print("5. kbs56774.csv vs current best")
print("  kbs shape:", kbs.shape)
# align by subject_id + lifelog_date
key = ["subject_id", "lifelog_date"]
m = best[key + TARGETS].merge(kbs[key + TARGETS], on=key, suffixes=("_best", "_kbs"))
print("  aligned rows:", len(m), "(expect 250)")
print("  per-target: kbs_mean | best_mean | mean|Δ| | max|Δ| | corr")
for t in TARGETS:
    a, b = m[f"{t}_kbs"], m[f"{t}_best"]
    d = (a - b).abs()
    print(f"    {t}: {a.mean():.4f} | {b.mean():.4f} | {d.mean():.4f} | {d.max():.4f} | {np.corrcoef(a,b)[0,1]:.4f}")
allbest = m[[f"{t}_best" for t in TARGETS]].values
allkbs = m[[f"{t}_kbs" for t in TARGETS]].values
print("  OVERALL mean|Δ|:", round(np.abs(allkbs-allbest).mean(), 5), "| max|Δ|:", round(np.abs(allkbs-allbest).max(), 5))
print("  identical to best?", np.allclose(allkbs, allbest, atol=1e-9))

# per-subject means for kbs (V131C cohort view)
print("=" * 70)
print("6. PER-SUBJECT label means (train) — cohort structure")
ps = train.groupby("subject_id")[TARGETS].mean().round(3)
print(ps.to_string())
print("  between-subject std (cohort spread):")
print("  ", {t: round(ps[t].std(), 3) for t in TARGETS})
