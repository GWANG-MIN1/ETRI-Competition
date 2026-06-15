#!/usr/bin/env python3
"""GOAL — MIS-LSTM-lite preprocessing: per-day multichannel BLOCK sequences for all subject-days.

Window 12:00 lifelog_date -> 12:00 next day (24h), 30-min blocks = 48 timesteps.
Channels per block: HR mean, HR std, steps sum, screen-on frac, charging frac, activity-still
frac, light mean (log). Built for ALL train rows (450). Saved as a tensor for the model script.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\박광민\Documents\Codex\etri_team\data")
DI = RAW / "ch2025_data_items"
HERE = Path(__file__).resolve().parent
BLOCK = 30          # minutes per block
NB = 48             # 24h / 30min
CH = ["hr_mean", "hr_std", "steps", "screen", "charge", "still", "light"]


def load(name, cols):
    d = pd.read_parquet(DI / f"ch2025_{name}.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d


print("loading sensors ...")
hr = load("wHr", None)[["subject_id", "timestamp", "heart_rate"]]
# heart_rate is an array per minute -> mean per minute
hr["hr_m"] = hr["heart_rate"].apply(lambda a: float(np.mean(a)) if a is not None and len(a) else np.nan)
hr = hr[["subject_id", "timestamp", "hr_m"]]
ped = load("wPedo", None)[["subject_id", "timestamp", "step"]]
scr = load("mScreenStatus", None)[["subject_id", "timestamp", "m_screen_use"]]
ac = load("mACStatus", None)[["subject_id", "timestamp", "m_charging"]]
act = load("mActivity", None)[["subject_id", "timestamp", "m_activity"]]
lig = load("mLight", None)[["subject_id", "timestamp", "m_light"]]
print("loaded.")

for d in (hr, ped, scr, ac, act, lig):
    d["tmin"] = d["timestamp"].dt.floor("1min")

tr = pd.read_csv(RAW / "ch2026_metrics_train.csv")
tr["lifelog_date"] = pd.to_datetime(tr["lifelog_date"])
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def block_feats(s, d):
    w0 = d + pd.Timedelta(hours=12); w1 = d + pd.Timedelta(days=1, hours=12)
    edges = [w0 + pd.Timedelta(minutes=BLOCK * i) for i in range(NB + 1)]
    X = np.full((NB, len(CH)), np.nan)
    def slice_s(df, col):
        x = df[(df.subject_id == s) & (df.tmin >= w0) & (df.tmin < w1)]
        return x
    H = slice_s(hr, "hr_m"); P = slice_s(ped, "step"); S = slice_s(scr, "m_screen_use")
    A = slice_s(ac, "m_charging"); AC = slice_s(act, "m_activity"); L = slice_s(lig, "m_light")
    any_data = False
    for i in range(NB):
        b0, b1 = edges[i], edges[i + 1]
        h = H[(H.tmin >= b0) & (H.tmin < b1)]["hr_m"].values
        p = P[(P.tmin >= b0) & (P.tmin < b1)]["step"].values
        sc = S[(S.tmin >= b0) & (S.tmin < b1)]["m_screen_use"].values
        ch = A[(A.tmin >= b0) & (A.tmin < b1)]["m_charging"].values
        av = AC[(AC.tmin >= b0) & (AC.tmin < b1)]["m_activity"].values
        lt = L[(L.tmin >= b0) & (L.tmin < b1)]["m_light"].values
        if len(h): X[i, 0] = np.nanmean(h); X[i, 1] = np.nanstd(h); any_data = True
        if len(p): X[i, 2] = np.nansum(p); any_data = True
        if len(sc): X[i, 3] = np.nanmean(sc); any_data = True
        if len(ch): X[i, 4] = np.nanmean(ch); any_data = True
        if len(av): X[i, 5] = np.mean(av == 3); any_data = True   # 3=STILL frac
        if len(lt): X[i, 6] = np.log1p(np.nanmean(lt)); any_data = True
    return X if any_data else None


print("building block sequences for 450 train rows ...")
seqs = []; ys = []; subs = []; mask = []
sid_map = {s: i for i, s in enumerate(sorted(tr.subject_id.unique()))}
for _, r in tr.iterrows():
    X = block_feats(r["subject_id"], r["lifelog_date"])
    if X is None:
        X = np.full((NB, len(CH)), np.nan); mask.append(0)
    else:
        mask.append(1)
    seqs.append(X); ys.append([r[t] for t in TARGETS]); subs.append(sid_map[r["subject_id"]])
X = np.array(seqs, dtype=np.float32)   # [N, NB, CH]
Y = np.array(ys, dtype=np.float32)
SUB = np.array(subs, dtype=np.int64)
M = np.array(mask)
print(f"X {X.shape}  Y {Y.shape}  usable rows {M.sum()}/{len(M)}")
np.savez(HERE / "mislstm_data.npz", X=X, Y=Y, SUB=SUB, keys=tr[["subject_id", "sleep_date", "lifelog_date"]].astype(str).values, CH=CH, TARGETS=TARGETS)
print("saved mislstm_data.npz")
