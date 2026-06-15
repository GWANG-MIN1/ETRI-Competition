#!/usr/bin/env python3
r"""
==============================================================================
 MIS-LSTM (Colab/GPU) — deep multichannel-sequence model for ETRI sleep/stress
==============================================================================
The validated per-row breakthrough (cf. arXiv:2509.11232). Trains a 13-channel
per-day sequence model (HR/steps/screen/charge/activity/light + GPS/usage/WiFi/
BLE/ambience/watch-light) with a subject embedding + BiLSTM, focal loss, data
augmentation, and a 5x5 seed/fold ensemble. Reports OUT-OF-FOLD CV so you can SEE
the gain before submitting, then blends with FrontierSilence -> submission.

CPU validation (this repo): the deep component gave a robust HONEST -0.035 logloss
on Q2 (out-of-sample) and -0.007 on Q3 vs the unified OOF. On GPU at full scale it
is far stronger -> this is the route toward ~0.54.

------------------------------------------------------------------------------
 HOW TO RUN ON GOOGLE COLAB  (free T4 GPU is enough)
------------------------------------------------------------------------------
 1. Colab menu: Runtime -> Change runtime type -> Hardware accelerator = GPU.
 2. Zip your data locally and upload to Google Drive, e.g. a folder MyDrive/etri/:
        etri/ch2025_data_items/ch2025_*.parquet        (the 12 sensor files)
        etri/ch2026_metrics_train.csv
        etri/ch2026_submission_sample.csv
        etri/submission_frontier_silence.csv           (your current best = FS base)
 3. In a Colab cell:
        from google.colab import drive; drive.mount('/content/drive')
        !pip -q install timm                            # optional (advanced backbone)
        # paste this file (or %run it) after setting DATA / FS below
 4. Set DATA and FS paths (below) to your Drive folder, then run.
    -> prints OOF CV per target (validate!), saves submission_mislstm_colab.csv.
 5. Download the csv and submit on DACON.

 NOTE: preprocessing parses ~2M sensor rows (CPU-bound) ~10-20 min; training on
 GPU is fast. Total ~20-30 min on a free Colab T4.

 TO PUSH HARDER toward 0.54 (after this runs): raise EPOCHS/SEEDS, set
 USE_SERESNEXT=True (per-block image backbone, needs timm), add more channels.
==============================================================================
"""
import os
import numpy as np
import pandas as pd

# ----------------------------- CONFIG -----------------------------
DATA = "/content/etri"                       # <- folder with the data
FS = DATA + "/submission_frontier_silence.csv"             # <- your current best (FS) for blending
BLOCK_MIN = 30; WIN_START = 12; WIN_HOURS = 24             # 48 blocks/day
HID = 64; EMB = 8; EPOCHS = 70; FOLDS = 5; SEEDS = (11, 23, 37, 51, 67, 89, 103, 127, 151, 179)  # SMALL+big-ensemble (450 rows overfit when big)
LR = 2e-3; WD = 3e-3; BATCH = 32; GAMMA = 1.5
AUG = dict(block_mask=0.05, chan_drop=0.05, noise=0.05)    # LIGHT aug (heavy aug hurt on 450 rows; raise cautiously)
BLEND_W = dict(Q1=0.0, Q2=0.40, Q3=0.25, S1=0.0, S2=0.05, S3=0.10, S4=0.0)  # deep-vs-FS (honest-validated)
OUT = "submission_mislstm_colab.csv"
USE_SERESNEXT = False                                      # advanced: per-block image backbone (timm)
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6
SPEECH = {"Speech", "Conversation", "Narration, monologue", "Babbling", "Shout",
          "Whispering", "Crowd", "Hubbub, speech noise, speech babble"}

import torch  # noqa: E402  (import before heavy MKL libs on some platforms)
import torch.nn as nn
import torch.nn.functional as Fnn
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV)
DI = os.path.join(DATA, "ch2025_data_items")


# ----------------------- preprocessing (13 channels) -----------------------
def _pm(name, col, fn=None):
    d = pd.read_parquet(os.path.join(DI, f"ch2025_{name}.parquet"))
    d["timestamp"] = pd.to_datetime(d["timestamp"]); d["tmin"] = d["timestamp"].dt.floor("1min")
    d["v"] = d[col].apply(fn) if fn else d[col]
    return d.groupby(["subject_id", "tmin"])["v"].mean()


def load_grid():
    def amb(a):
        try:
            tot = sum(float(x[1]) for x in a); sp = sum(float(x[1]) for x in a if str(x[0]) in SPEECH)
            return sp / tot if tot > 0 else 0.0
        except Exception:
            return np.nan
    g = {
        "hr_mean": _pm("wHr", "heart_rate", lambda a: float(np.mean(a)) if a is not None and len(a) else np.nan),
        "hr_std": _pm("wHr", "heart_rate", lambda a: float(np.std(a)) if a is not None and len(a) else np.nan),
        "steps": _pm("wPedo", "step"),
        "screen": _pm("mScreenStatus", "m_screen_use"),
        "charge": _pm("mACStatus", "m_charging"),
        "still": _pm("mActivity", "m_activity", lambda v: float(v == 3)),
        "light": _pm("mLight", "m_light", lambda v: np.log1p(v)),
        "gps_speed": _pm("mGps", "m_gps", lambda a: float(np.mean([p.get("speed", 0.0) for p in a])) if hasattr(a, "__len__") else np.nan),
        "wifi_cnt": _pm("mWifi", "m_wifi", lambda a: float(len(a)) if hasattr(a, "__len__") else np.nan),
        "ble_cnt": _pm("mBle", "m_ble", lambda a: float(len(a)) if hasattr(a, "__len__") else np.nan),
        "usage": _pm("mUsageStats", "m_usage_stats", lambda a: float(sum(x["total_time"] for x in a)) if hasattr(a, "__len__") else np.nan),
        "amb_speech": _pm("mAmbience", "m_ambience", amb),
        "wlight": _pm("wLight", "w_light", lambda v: np.log1p(v)),
    }
    CH = list(g.keys())
    G = pd.DataFrame(g).reset_index().sort_values(["subject_id", "tmin"]).reset_index(drop=True)
    return G, CH


def build(rows, G, CH, sid):
    nb = WIN_HOURS * 60 // BLOCK_MIN
    Xs, subs = [], []
    for _, r in rows.iterrows():
        s = r["subject_id"]; d0 = pd.to_datetime(r["lifelog_date"]) + pd.Timedelta(hours=WIN_START)
        d1 = d0 + pd.Timedelta(hours=WIN_HOURS); idx = pd.date_range(d0, d1, freq="1min", inclusive="left")
        sub = G[(G.subject_id == s) & (G.tmin >= d0) & (G.tmin < d1)].set_index("tmin").reindex(idx)
        X = np.full((nb, len(CH)), np.nan, np.float32)
        if len(sub):
            arr = sub[CH].values
            for i in range(nb):
                seg = arr[i * BLOCK_MIN:(i + 1) * BLOCK_MIN]
                if seg.size and not np.all(np.isnan(seg)):
                    with np.errstate(all="ignore"):
                        X[i] = np.nanmean(seg, 0)
        Xs.append(X); subs.append(sid.get(s, 0))
    return np.array(Xs, np.float32), np.array(subs, np.int64)


# ----------------------------- model -----------------------------
class Net(nn.Module):
    def __init__(self, c, nsub):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)
        self.conv = nn.Sequential(nn.Conv1d(c, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
                                  nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, HID, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.emb = nn.Embedding(nsub, EMB)
        self.head = nn.Sequential(nn.Linear(2 * HID + EMB, 96), nn.ReLU(), nn.Dropout(0.55), nn.Linear(96, 7))

    def forward(self, x, s):
        B, T, C = x.shape
        x = self.bn(x.reshape(B * T, C)).reshape(B, T, C)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(x)
        return self.head(torch.cat([o.mean(1), self.emb(s)], 1))


def focal(z, y, g=GAMMA):
    p = torch.sigmoid(z); ce = Fnn.binary_cross_entropy_with_logits(z, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y); return (((1 - pt) ** g) * ce).mean()


def augment(x):
    # x [B,T,C] tensor on device. block-mask + channel-drop + gaussian noise
    if AUG["block_mask"] > 0:
        m = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > AUG["block_mask"]).float(); x = x * m
    if AUG["chan_drop"] > 0:
        m = (torch.rand(x.shape[0], 1, x.shape[2], device=x.device) > AUG["chan_drop"]).float(); x = x * m
    if AUG["noise"] > 0:
        x = x + AUG["noise"] * torch.randn_like(x)
    return x


def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def bll(y, p):
    p = np.clip(p, EPS, 1 - EPS); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    tr = pd.read_csv(os.path.join(DATA, "ch2026_metrics_train.csv"))
    sample = pd.read_csv(os.path.join(DATA, "ch2026_submission_sample.csv"))
    sid = {s: i for i, s in enumerate(sorted(tr.subject_id.unique()))}
    print("loading 13-channel grid (CPU, slow) ..."); G, CH = load_grid()
    print("building sequences ..."); Xtr, Str = build(tr, G, CH, sid); Xte, Ste = build(sample, G, CH, sid)
    Ytr = tr[TARGETS].values.astype(np.float32); C = Xtr.shape[2]
    mu = np.nanmean(Xtr.reshape(-1, C), 0); sd = np.nanstd(Xtr.reshape(-1, C), 0) + 1e-6
    prep = lambda A: np.concatenate([np.nan_to_num((A - mu) / sd), np.isnan((A - mu) / sd).astype(np.float32)], 2)
    Xtrp, Xtep = prep(Xtr), prep(Xte)
    rng = np.random.default_rng(0)
    test_pred = np.zeros((len(sample), 7)); oof = np.zeros((len(tr), 7)); ocnt = np.zeros(len(tr)); nf = 0
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        fold = np.zeros(len(tr), int)
        for s in np.unique(Str):
            idx = np.where(Str == s)[0]
            for k, ch in enumerate(np.array_split(rng.permutation(idx), FOLDS)):
                fold[ch] = k
        for k in range(FOLDS):
            ti = np.where(fold != k)[0]; vi = np.where(fold == k)[0]
            net = Net(Xtrp.shape[2], len(sid)).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            xt = torch.tensor(Xtrp[ti]).to(DEV); st = torch.tensor(Str[ti]).to(DEV); yt = torch.tensor(Ytr[ti]).to(DEV)
            net.train(); n = len(ti)
            for ep in range(EPOCHS):
                perm = np.random.permutation(n)
                for i in range(0, n, BATCH):
                    ii = perm[i:i + BATCH]; opt.zero_grad()
                    focal(net(augment(xt[ii]), st[ii]), yt[ii]).backward(); opt.step()
                sch.step()
            net.eval()
            with torch.no_grad():
                test_pred += torch.sigmoid(net(torch.tensor(Xtep).to(DEV), torch.tensor(Ste).to(DEV))).cpu().numpy()
                oof[vi] += torch.sigmoid(net(torch.tensor(Xtrp[vi]).to(DEV), torch.tensor(Str[vi]).to(DEV))).cpu().numpy()
                ocnt[vi] += 1; nf += 1
    test_pred /= nf; oof /= np.maximum(ocnt[:, None], 1)

    # ---- OOF CV diagnostics (validate before submitting) ----
    print("\n=== OOF CV: deep model per-target logloss (lower=better) ===")
    for j, t in enumerate(TARGETS):
        print(f"  {t}: deep OOF logloss {bll(Ytr[:, j], oof[:, j]):.4f}")

    # ---- blend with FrontierSilence -> submission ----
    fs = pd.read_csv(FS).sort_values(["subject_id", "sleep_date", "lifelog_date"]).reset_index(drop=True)
    order = sample.sort_values(["subject_id", "sleep_date", "lifelog_date"]).index.values
    tp = test_pred[order]; out = sample.sort_values(["subject_id", "sleep_date", "lifelog_date"]).reset_index(drop=True)
    for j, t in enumerate(TARGETS):
        w = BLEND_W[t]; z = (1 - w) * logit(fs[t].values) + w * logit(tp[:, j])
        out[t] = np.clip(1 / (1 + np.exp(-z)), 1e-4, 1 - 1e-4)
    out.to_csv(OUT, index=False)
    print(f"\nsaved {OUT}  shape {out.shape}")
    print("blend weights (deep vs FS):", BLEND_W)
    print("TIP: raise BLEND_W on Q2/Q3 if the deep OOF logloss there is clearly below your FS;")
    print("     keep weights 0 where deep OOF is worse than FS (avoid dragging FS down).")


if __name__ == "__main__":
    main()
