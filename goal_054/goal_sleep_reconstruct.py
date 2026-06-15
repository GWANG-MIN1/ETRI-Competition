#!/usr/bin/env python3
"""GOAL pivot — reconstruct objective sleep params from PHONE sensors (97% night coverage).

S1-S4 are deterministic NSF-threshold functions of TST/SE/SOL/WASO measured by a Withings bed
sensor (not in the data). Reconstruct proxies from phone screen/activity/charging:
  main sleep period = longest contiguous screen-OFF + stationary block in the night window.
  TST   = asleep minutes;  TIB = bedtime..final-wake;  SE = TST/TIB;
  SOL   = bedtime..sleep-onset;  WASO = awake minutes inside [onset, final-wake].
Validate against train S labels: AUC of each proxy vs its S target, and the best-threshold logloss.
If proxies discriminate S well, this is the 0.54 paradigm (measurement, not noisy prediction).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\박광민\Documents\Codex\etri_team\data")
DI = RAW / "ch2025_data_items"
EPS = 1e-6


def load(name, col):
    d = pd.read_parquet(DI / f"ch2025_{name}.parquet")[["subject_id", "timestamp", col]]
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d


scr = load("mScreenStatus", "m_screen_use")
act = load("mActivity", "m_activity")
acc = load("mACStatus", "m_charging")
print("loaded screen/activity/charging")

tr = pd.read_csv(RAW / "ch2026_metrics_train.csv")
tr["lifelog_date"] = pd.to_datetime(tr["lifelog_date"])


def night_features(s, d):
    """reconstruct sleep features for subject s, lifelog night starting on date d."""
    w0 = d + pd.Timedelta(hours=18)
    w1 = d + pd.Timedelta(days=1, hours=14)
    idx = pd.date_range(w0, w1, freq="1min", inclusive="left")
    # per-minute screen-on (1 if any on), charging frac, activity mode
    def grid(df, col, agg="max"):
        x = df[(df.subject_id == s) & (df.timestamp >= w0) & (df.timestamp < w1)]
        if len(x) == 0:
            return pd.Series(np.nan, index=idx)
        g = x.set_index("timestamp")[col].resample("1min").agg(agg)
        return g.reindex(idx)
    screen = grid(scr, "m_screen_use", "max")          # 1=on
    charge = grid(acc, "m_charging", "max")             # 1=charging
    activ = grid(act, "m_activity", lambda v: v.mode().iloc[0] if len(v) else np.nan)
    n = len(idx)
    have = (~screen.isna()).sum()
    if have < 120:
        return None
    screen = screen.fillna(0).values
    charge = charge.fillna(0).values
    activ = activ.fillna(-1).values
    # asleep-candidate minute: screen off
    asleep = (screen == 0).astype(int)
    # find longest contiguous asleep run, merging gaps <= 15 min (brief wakes)
    best = (0, 0, 0)  # len, start, end
    i = 0
    runs = []
    j = 0
    while j < n:
        if asleep[j] == 1:
            k = j
            gap = 0
            while k + 1 < n and (asleep[k + 1] == 1 or gap < 15):
                if asleep[k + 1] == 0:
                    gap += 1
                else:
                    gap = 0
                k += 1
            runs.append((j, k))
            j = k + 1
        else:
            j += 1
    if not runs:
        return None
    # main sleep = longest run, restricted to start after 20:00 (avoid evening)
    runs2 = [(a, b) for (a, b) in runs if (b - a) >= 90]
    if not runs2:
        runs2 = runs
    a, b = max(runs2, key=lambda r: r[1] - r[0])
    onset_min, wake_min = a, b
    # TST = asleep minutes within [onset, wake]; WASO = awake (screen-on) minutes inside
    seg = slice(onset_min, wake_min + 1)
    tib = (wake_min - onset_min)
    waso = int((screen[seg] == 1).sum())
    tst = tib - waso
    se = tst / max(tib, 1)
    # bedtime: first charging-start or sustained screen-off in the 2h before onset
    pre = slice(max(0, onset_min - 120), onset_min + 1)
    bed = onset_min
    ch_pre = np.where(charge[pre] == 1)[0]
    if len(ch_pre):
        bed = max(0, onset_min - 120) + ch_pre[0]
    sol = max(0, onset_min - bed)
    return dict(tst=tst, tib=tib, se=se, sol=sol, waso=waso,
                onset_h=(onset_min / 60.0 + 18) % 24, cover=have)


print("reconstructing 450 train nights ...")
feats = []
for _, r in tr.iterrows():
    f = night_features(r["subject_id"], r["lifelog_date"])
    feats.append(f if f else {})
F = pd.DataFrame(feats)
tr2 = pd.concat([tr.reset_index(drop=True), F], axis=1)
ok = tr2["tst"].notna()
print(f"reconstructed {ok.sum()}/450 nights (>=120 min phone coverage)\n")


def auc(y, x):
    from scipy.stats import rankdata
    y = np.asarray(y); x = np.asarray(x)
    m = ~np.isnan(x)
    y, x = y[m], x[m]
    if len(np.unique(y)) < 2:
        return np.nan
    r = rankdata(x)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# proxy -> target mapping (NSF direction): S1~TST (higher=better->1), S2~SE(higher), S3~SOL(lower better),
# S4~WASO(lower better)
print("=== reconstructed proxy vs S target (AUC; >0.5 means proxy discriminates the binary) ===")
print(f"  {'target':>6} | {'proxy':>5} | {'AUC':>6} | {'note'}")
pairs = [("S1", "tst", +1, "TST higher -> recommended"),
         ("S2", "se", +1, "SE higher -> recommended"),
         ("S3", "sol", -1, "SOL lower -> recommended"),
         ("S4", "waso", -1, "WASO lower -> recommended")]
for tgt, prx, sgn, note in pairs:
    a = auc(tr2.loc[ok, tgt].values, sgn * tr2.loc[ok, prx].values)
    print(f"  {tgt:>6} | {prx:>5} | {a:>6.3f} | {note}")

print("\n=== per-night TST distribution (sanity: hours) ===")
print((tr2.loc[ok, "tst"] / 60.0).describe().round(2).to_string())
print("\n  AUC >> 0.5 on S1/S2 would confirm reconstruction carries strong, transferable per-row signal.")
tr2.to_parquet(Path(__file__).resolve().parent / "sleep_recon_train.parquet")
print("  saved sleep_recon_train.parquet")
