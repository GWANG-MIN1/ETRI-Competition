#!/usr/bin/env python3
"""GOAL — rich-channel MIS-LSTM preprocessing (13 channels per the paper's important modalities).

Adds to the 7 base channels: gps_speed, wifi_count, ble_count, usage_time, ambience-speech frac,
watch-light. Window 12:00->12:00 next day, 30-min blocks (48). Saves mislstm_data_v3.npz.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\박광민\Documents\Codex\etri_team\data")
DI = RAW / "ch2025_data_items"
HERE = Path(__file__).resolve().parent
NB = 48; BLOCK = 30
SPEECH = {"Speech", "Conversation", "Narration, monologue", "Babbling", "Speech synthesizer",
          "Shout", "Whispering", "Crowd", "Hubbub, speech noise, speech babble"}


def pm(name, fn, col):
    d = pd.read_parquet(DI / f"ch2025_{name}.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"]); d["tmin"] = d["timestamp"].dt.floor("1min")
    d["v"] = d[col].apply(fn)
    return d.groupby(["subject_id", "tmin"])["v"].mean()


print("parsing rich modalities to per-minute (slow) ...")
def gps_speed(a):
    try:
        return float(np.mean([p.get("speed", 0.0) for p in a]))
    except Exception:
        return np.nan
def amb_speech(a):
    try:
        tot = 0.0; sp = 0.0
        for x in a:
            lab = str(x[0]); pr = float(x[1]); tot += pr
            if lab in SPEECH:
                sp += pr
        return sp / tot if tot > 0 else 0.0
    except Exception:
        return np.nan

ch = {}
ch["gps_speed"] = pm("mGps", gps_speed, "m_gps");                         print(" gps done")
ch["wifi_cnt"] = pm("mWifi", lambda a: float(len(a)) if hasattr(a, "__len__") else np.nan, "m_wifi"); print(" wifi done")
ch["ble_cnt"] = pm("mBle", lambda a: float(len(a)) if hasattr(a, "__len__") else np.nan, "m_ble"); print(" ble done")
ch["usage"] = pm("mUsageStats", lambda a: float(sum(x["total_time"] for x in a)) if hasattr(a, "__len__") else np.nan, "m_usage_stats"); print(" usage done")
ch["amb_speech"] = pm("mAmbience", amb_speech, "m_ambience");             print(" ambience done")
wl = pd.read_parquet(DI / "ch2025_wLight.parquet"); wl["timestamp"] = pd.to_datetime(wl["timestamp"]); wl["tmin"] = wl["timestamp"].dt.floor("1min")
ch["wlight"] = wl.groupby(["subject_id", "tmin"])["w_light"].mean().apply(np.log1p); print(" wlight done")

# combine into one per-minute frame
grid = pd.DataFrame(ch).reset_index()
RICH = ["gps_speed", "wifi_cnt", "ble_cnt", "usage", "amb_speech", "wlight"]
grid = grid.sort_values(["subject_id", "tmin"]).reset_index(drop=True)

tr = pd.read_csv(RAW / "ch2026_metrics_train.csv"); tr["lifelog_date"] = pd.to_datetime(tr["lifelog_date"])

def rich_blocks(s, d):
    w0 = d + pd.Timedelta(hours=12); w1 = d + pd.Timedelta(days=1, hours=12)
    idx = pd.date_range(w0, w1, freq="1min", inclusive="left")
    sub = grid[(grid.subject_id == s) & (grid.tmin >= w0) & (grid.tmin < w1)].set_index("tmin")
    X = np.full((NB, len(RICH)), np.nan, np.float32)
    if len(sub) == 0:
        return X
    sub = sub.reindex(idx)
    for i in range(NB):
        seg = sub.iloc[i * BLOCK:(i + 1) * BLOCK]
        for j, c in enumerate(RICH):
            v = seg[c].values
            if np.any(~np.isnan(v)):
                X[i, j] = np.nanmean(v)
    return X


print("building rich block sequences for 450 rows ...")
Xr = np.array([rich_blocks(r["subject_id"], r["lifelog_date"]) for _, r in tr.iterrows()], np.float32)
# load base 7-channel and concat
base = np.load(HERE / "mislstm_data.npz", allow_pickle=True)
Xb = base["X"]
X = np.concatenate([Xb, Xr], axis=2)  # [N,48,13]
print(f"combined X {X.shape}")
np.savez(HERE / "mislstm_data_v3.npz", X=X, Y=base["Y"], SUB=base["SUB"], keys=base["keys"],
         CH=list(base["CH"]) + RICH, TARGETS=base["TARGETS"])
print("saved mislstm_data_v3.npz with channels:", list(base["CH"]) + RICH)
