"""Strategy-doc Step 1 DIAGNOSTIC: is there room for an S2/S4 sleep-proxy specialist,
and does the existing 5242-feature base already subsume night/sleep features?

Decisive gate before building anything: if base already contains night-window sleep
proxies, a new proxy cannot help (subsumed). Prints (a) feature-store night/sleep column
inventory, (b) canvas night-window sanity, (c) per-target base room.
NO LB fitting. Pure offline diagnostic on the 450 train rows.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

KIT_DIR = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
FS_PARQUET = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\outputs\features\feature_store.parquet")
CANVAS = Path(__file__).resolve().parents[1] / "worldmodel_jepa" / "cache" / "wm_canvas.npz"
sys.path.insert(0, str(HSJEPA_SRC)); sys.path.insert(0, str(KIT_DIR))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa

# ---- (a) feature store night/sleep inventory ----
fs = pd.read_parquet(FS_PARQUET)
cols = [c for c in fs.columns if c not in H.KEYS]
print(f"feature store: {fs.shape}  feature cols={len(cols)}")
kw = ["night","sleep","bed","wake","waso","onset","hr","rmssd","hrv","light","screen",
      "unlock","charg","motion","step","speed","activ","ambien","wifi","ble","nocturn",
      "rest","quiet","inactiv","dur","hour","time","circad","dawn","dusk","22","23","00","01","02","03","04","05","06"]
buckets = {k: [c for c in cols if k.lower() in c.lower()] for k in kw}
print("\n-- feature-store keyword hits (count) --")
for k, v in buckets.items():
    if v:
        print(f"  {k:10s}: {len(v):4d}   e.g. {v[:3]}")
# explicit night-window detection: any column referencing a clock hour in 21-07 range?
import re
night_like = [c for c in cols if re.search(r"(night|sleep|bed|wake|waso|onset|nocturn|inactiv)", c, re.I)]
print(f"\nexplicit night/sleep-named columns: {len(night_like)}")
for c in night_like[:40]:
    print("   ", c)

# ---- (b) canvas night-window sanity ----
d = np.load(CANVAS, allow_pickle=True)
X = d["X"]; obs = d["obs"]; chans = list(d["chans"]); n_tr = int(d["n_tr"]); T = int(d["T"])
print(f"\ncanvas X={X.shape} obs={obs.shape} n_tr={n_tr} T={T}")
print("chans:", chans)
# bins: anchor 12:00, 5-min. hour h in [12,36) -> bin=(h-12)*12
def hb(h):  # clock hour -> bin
    hh = h if h >= 12 else h + 24
    return int((hh - 12) * 12)
night = slice(hb(22), hb(9 + 24) if False else hb(33))  # 22:00 .. 09:00
print(f"night bins {night.start}..{night.stop}  (22:00->09:00)")
# coverage in night window per channel (train rows)
for ci, ch in enumerate(chans):
    cov = obs[:n_tr, ci, night].mean()
    print(f"  ch{ci:2d} {ch:12s} night-coverage(train)={cov:.3f}")

# ---- (c) per-target base room ----
m = K.base()
base_arr = np.column_stack([m[f"p_{t}"].values for t in K.TARGETS])
sc = K.score_perrow(m, base_arr)
print("\n-- base test_faithful per-target & per-subject-constant floor --")
# per-subject TRUE-rate constant (oracle level floor; leaky, only to bound room)
for t in K.TARGETS:
    y = m[f"y_{t}"].values
    # oracle per-subject constant prediction
    psubj = np.zeros(len(m))
    for s in m["subject_id"].unique():
        mk = m["subject_id"].values == s
        psubj[mk] = y[mk].mean()
    # global constant
    pg = np.full(len(m), y.mean())
    # honest scores (note psubj/pg are leaky oracles, only as reference bounds)
    ll_base = sc[t]
    ll_subj = H.bll(y, np.clip(psubj, 1e-6, 1-1e-6))
    ll_glob = H.bll(y, np.clip(pg, 1e-6, 1-1e-6))
    print(f"  {t}: base={ll_base:.4f}  |oracle subj-const={ll_subj:.4f}  glob-const={ll_glob:.4f}"
          f"  (positives={y.mean():.3f})")
print("\nDONE")
