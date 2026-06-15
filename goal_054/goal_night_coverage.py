#!/usr/bin/env python3
"""GOAL — cheap gate before the heavy FM test: does night-window HR coverage even exist?

The FM-embedding paradigm (and any night-physiology paradigm) needs a per-minute HR series
in each row's sleep window. The physio assessment claimed nocturnal coverage collapses.
Verify directly: for each LABELED train row (subject, lifelog_date), how many minutes of wHr
fall in the night window 22:00(prev)–09:00? If most rows have ~0, the paradigm is dead on
arrival (no substrate), and the heavy FM install is pointless.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\박광민\Documents\Codex\etri_team\data")
hr = pd.read_parquet(RAW / "ch2025_data_items" / "ch2025_wHr.parquet")
hr["timestamp"] = pd.to_datetime(hr["timestamp"])
lbl = pd.read_csv(RAW / "ch2026_metrics_train.csv")
lbl["lifelog_date"] = pd.to_datetime(lbl["lifelog_date"])

# minutes of HR per (subject, calendar date) and per night window
hr["date"] = hr["timestamp"].dt.normalize()
hr["hour"] = hr["timestamp"].dt.hour
# night = 22:00..23:59 of lifelog_date + 00:00..08:59 of next day
print("=== HR minute coverage per labeled row's NIGHT window (22:00 lifelog -> 09:00 next) ===")
counts = []
night_counts = []
for _, r in lbl.iterrows():
    s = r["subject_id"]; d = r["lifelog_date"]
    sub = hr[hr["subject_id"] == s]
    # full lifelog-date day
    day = sub[(sub["timestamp"] >= d) & (sub["timestamp"] < d + pd.Timedelta(days=1))]
    counts.append(len(day))
    # night window
    start = d + pd.Timedelta(hours=22)
    end = d + pd.Timedelta(days=1, hours=9)
    night = sub[(sub["timestamp"] >= start) & (sub["timestamp"] < end)]
    # also early-morning core 00-06h of next day
    core = sub[(sub["timestamp"] >= d + pd.Timedelta(days=1)) & (sub["timestamp"] < d + pd.Timedelta(days=1, hours=6))]
    night_counts.append((len(night), len(core)))

counts = np.array(counts)
nc = np.array([x[0] for x in night_counts]); cc = np.array([x[1] for x in night_counts])
N = len(lbl)
print(f"  labeled rows: {N}")
print(f"  full-day HR minutes:   median {np.median(counts):.0f}  mean {counts.mean():.0f}  rows>0: {int((counts>0).sum())}/{N}")
print(f"  night(22-09) minutes:  median {np.median(nc):.0f}  mean {nc.mean():.0f}  rows>=60min: {int((nc>=60).sum())}/{N} ({(nc>=60).mean()*100:.0f}%)")
print(f"  core(00-06) minutes:   median {np.median(cc):.0f}  mean {cc.mean():.0f}  rows>=60min: {int((cc>=60).sum())}/{N} ({(cc>=60).mean()*100:.0f}%)")
print()
print(f"  rows with usable night series (>=120 night-min): {int((nc>=120).sum())}/{N} ({(nc>=120).mean()*100:.0f}%)")
print(f"  rows with ZERO night HR: {int((nc==0).sum())}/{N} ({(nc==0).mean()*100:.0f}%)")
print()
if (nc >= 120).mean() < 0.5:
    print("VERDICT: night-HR substrate is SPARSE (<50% of rows usable) -> a night-window FM/physio")
    print("  per-row signal cannot transfer (most rows have no substrate). Heavy FM install not justified")
    print("  for a per-row gain; coverage caps the achievable transferable gain far below the 0.009 floor.")
else:
    print("VERDICT: night-HR substrate is ADEQUATE -> proceed to the frozen-FM GATE A/B/C test.")
