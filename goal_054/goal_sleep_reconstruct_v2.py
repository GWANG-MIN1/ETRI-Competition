#!/usr/bin/env python3
"""GOAL pivot v2 — better phone-based sleep reconstruction + DECISIVE comparison to the team model.

Pre-resample sensors to per-minute per-subject grids (fast). Sleep period = longest nocturnal
block of (screen-off & no-steps), refined by overnight charging (TIB). Compute TST/SE/SOL/WASO.
Then the DECISIVE test: AUC of each reconstructed proxy vs S target, side by side with the team's
unified-OOF model AUC on the SAME rows. Reconstruction only opens a path to 0.54 if it BEATS or
ADDS-orthogonally to the team model on the duration-based targets (S1 TST / S2 SE).
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\박광민\Documents\Codex\etri_team\data")
DI = RAW / "ch2025_data_items"
CORE = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(CORE))
import hsjepa_core as H  # noqa: E402


def per_min(name, col, agg):
    d = pd.read_parquet(DI / f"ch2025_{name}.parquet")[["subject_id", "timestamp", col]]
    d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.floor("1min")
    g = d.groupby(["subject_id", "timestamp"])[col].agg(agg)
    return g


print("pre-resampling sensors to per-minute ...")
screen = per_min("mScreenStatus", "m_screen_use", "max")   # 1=on
charge = per_min("mACStatus", "m_charging", "max")          # 1=charging
steps = per_min("wPedo", "step", "sum")                     # steps that minute
G = pd.DataFrame({"screen": screen, "charge": charge, "steps": steps}).reset_index()
G = G.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)
print(f"grid rows {len(G)}")

tr = pd.read_csv(RAW / "ch2026_metrics_train.csv")
tr["lifelog_date"] = pd.to_datetime(tr["lifelog_date"])
tr["sleep_date"] = pd.to_datetime(tr["sleep_date"])


def reconstruct(s, d):
    w0 = d + pd.Timedelta(hours=20); w1 = d + pd.Timedelta(days=1, hours=12)
    idx = pd.date_range(w0, w1, freq="1min", inclusive="left")
    sub = G[(G.subject_id == s) & (G.timestamp >= w0) & (G.timestamp < w1)].set_index("timestamp")
    if len(sub) < 120:
        return None
    scr = sub["screen"].reindex(idx).fillna(0).values
    chg = sub["charge"].reindex(idx).fillna(0).values
    stp = sub["steps"].reindex(idx).fillna(0).values
    n = len(idx)
    # quiescent minute: screen off AND no steps
    quiet = ((scr == 0) & (stp == 0)).astype(int)
    # smooth: require quiet; find longest run allowing <=10min interruptions
    runs = []
    j = 0
    while j < n:
        if quiet[j]:
            k = j; gap = 0; last = j
            while k + 1 < n:
                if quiet[k + 1]:
                    last = k + 1; gap = 0
                else:
                    gap += 1
                    if gap > 10:
                        break
                k += 1
            runs.append((j, last)); j = last + 1
        else:
            j += 1
    runs = [(a, b) for (a, b) in runs if b - a >= 90]
    if not runs:
        return None
    a, b = max(runs, key=lambda r: r[1] - r[0])
    tib = b - a
    waso = int((scr[a:b + 1] == 1).sum() + (stp[a:b + 1] > 0).sum())
    tst = max(0, tib - waso)
    se = tst / max(tib, 1)
    # SOL: minutes from bedtime (first charge-on in 2h before onset, else onset) to onset
    pre = slice(max(0, a - 120), a + 1)
    chp = np.where(chg[pre] == 1)[0]
    bed = (max(0, a - 120) + chp[0]) if len(chp) else a
    sol = a - bed
    return dict(tst=tst, tib=tib, se=se, sol=sol, waso=waso)


print("reconstructing ...")
F = pd.DataFrame([reconstruct(r["subject_id"], r["lifelog_date"]) or {} for _, r in tr.iterrows()])
tr2 = pd.concat([tr.reset_index(drop=True), F], axis=1)
ok = tr2["tst"].notna().values
print(f"reconstructed {ok.sum()}/450; TST hours: mean {np.nanmean(tr2['tst'])/60:.2f} "
      f"median {np.nanmedian(tr2['tst'])/60:.2f} (target ~7)")

# team model OOF AUC baseline (merge on keys)
m = H.load()
mm = m.merge(tr2, on=["subject_id", "sleep_date", "lifelog_date"], how="inner", suffixes=("", "_r"))
okm = mm["tst"].notna().values


def auc(y, x):
    from scipy.stats import rankdata
    y = np.asarray(y, float); x = np.asarray(x, float)
    keep = ~np.isnan(x) & ~np.isnan(y)
    y, x = y[keep], x[keep]
    if len(np.unique(y)) < 2:
        return np.nan
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print("\n=== DECISIVE: reconstruction AUC vs TEAM-model (unified OOF) AUC, same rows ===")
print(f"  {'target':>6} | {'recon proxy':>11} | {'recon AUC':>9} | {'team OOF AUC':>12} | {'recon adds?':>11}")
pairs = [("S1", "tst", +1), ("S2", "se", +1), ("S3", "sol", -1), ("S4", "waso", -1)]
for tgt, prx, sgn in pairs:
    ra = auc(mm.loc[okm, f"y_{tgt}"].values, sgn * mm.loc[okm, prx].values)
    ta = auc(mm.loc[okm, f"y_{tgt}"].values, mm.loc[okm, f"p_{tgt}"].values)
    verdict = "BEATS team" if ra > ta + 0.02 else ("~ties" if abs(ra - ta) <= 0.02 else "loses")
    print(f"  {tgt:>6} | {prx:>11} | {ra:>9.3f} | {ta:>12.3f} | {verdict:>11}")

print("\n  If recon AUC >> team OOF AUC on S1/S2, reconstruction is a real new lever (build full model).")
print("  If recon ties/loses, the team feature store already captured it -> reconstruction is subsumed.")
