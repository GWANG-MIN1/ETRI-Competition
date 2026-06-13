"""Build HS-JEPA submission ladder from a recipe JSON, with geometry guard + 7-step checklist.

Recipe JSON:
{
  "rungs": {
     "conservative": {
        "global": {"Q2": +0.02, "S2": -0.02, ...},      # additive prob LEVEL shift on top of best
        "deshrink": [{"target":"S3","frac":0.25}, ...]   # move subj submission-mean FRAC toward its
                                                         # own train mean (capped, no overshoot) = V131C de-shrink
     },
     "moderate": {...}
  }
}
Only LEVEL moves. Geometry guard: clip per-subject to train range +/- margin, then global [0.02,0.98].
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBS = ROOT / "outputs" / "submissions"
RAW = ROOT / "outputs" / "raw"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
BEST = SUBS / "submission_FINAL_Q2recency_plus_Q3recency.csv"
MARGIN, LO, HI = 0.05, 0.02, 0.98


def subj_train_stats():
    lbl = pd.read_csv(RAW / "ch2026_metrics_train.csv")
    return (lbl.groupby("subject_id")[TARGETS].mean(),
            lbl.groupby("subject_id")[TARGETS].min(),
            lbl.groupby("subject_id")[TARGETS].max())


def apply_recipe(best, rung, mean, mn, mx):
    out = best.copy()
    for t, d in rung.get("global", {}).items():
        out[t] = out[t] + float(d)
    for op in rung.get("deshrink", []):
        t, frac = op["target"], float(op["frac"])
        for s in out["subject_id"].unique():
            msk = out["subject_id"] == s
            cur = out.loc[msk, t].mean()
            shift = frac * (mean.at[s, t] - cur)   # toward own train mean, capped at mean (no overshoot)
            out.loc[msk, t] = out.loc[msk, t] + shift
    for s in out["subject_id"].unique():           # geometry guard
        msk = out["subject_id"] == s
        for t in TARGETS:
            lo = max(LO, mn.at[s, t] - MARGIN); hi = min(HI, mx.at[s, t] + MARGIN)
            out.loc[msk, t] = out.loc[msk, t].clip(lo, hi)
    out[TARGETS] = out[TARGETS].clip(LO, HI)
    return out


def checklist(sub, best, name):
    print(f"\n===== CHECKLIST: {name} =====")
    sample = pd.read_csv(RAW / "ch2026_submission_sample.csv")
    c1 = sub.shape == (250, 10)
    c2 = list(sub.columns) == list(sample.columns)
    c3 = int(sub[TARGETS].isnull().sum().sum()) == 0
    c4 = bool(sub[TARGETS].min().min() >= 0 and sub[TARGETS].max().max() <= 1)
    print(f"  [1] shape(250,10): {sub.shape} -> {c1}")
    print(f"  [2] cols==sample: {c2}")
    print(f"  [3] null==0: {c3}")
    print(f"  [4] range[0,1]: [{sub[TARGETS].min().min():.4f},{sub[TARGETS].max().max():.4f}] -> {c4}")
    d = sub[TARGETS].values - best[TARGETS].values
    print(f"  [5] mean|diff| vs best={np.abs(d).mean():.4f} max|diff|={np.abs(d).max():.4f}")
    for t in TARGETS:
        print(f"        {t}: {best[t].mean():.4f} -> {sub[t].mean():.4f}  ({sub[t].mean()-best[t].mean():+.4f})")
    tmp = sub.copy(); tmp["lifelog_date"] = pd.to_datetime(tmp["lifelog_date"])
    sp = {t: float(tmp.groupby("lifelog_date")[t].mean().std()) for t in TARGETS}
    print(f"  [6] date-mean std: " + " ".join(f"{t}={sp[t]:.3f}" for t in TARGETS))
    ok = c1 and c2 and c3 and c4
    print(f"  [7] RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    recipe = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    best = pd.read_csv(BEST)
    mean, mn, mx = subj_train_stats()
    cols = list(pd.read_csv(RAW / "ch2026_submission_sample.csv").columns)
    for rname, rung in recipe["rungs"].items():
        sub = apply_recipe(best, rung, mean, mn, mx)
        if checklist(sub, best, rname):
            outp = SUBS / f"submission_HSJEPA_{rname}.csv"
            sub[cols].to_csv(outp, index=False)
            print(f"  SAVED -> {outp}")
        else:
            print("  NOT SAVED (checklist failed)")


if __name__ == "__main__":
    main()
