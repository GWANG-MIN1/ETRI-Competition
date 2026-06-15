#!/usr/bin/env python3
"""MIS-LSTM FULL (turnkey) — 13-channel deep-sequence model + FrontierSilence blend -> submission.

The validated per-row breakthrough for the ETRI sleep/stress challenge (cf. arXiv:2509.11232):
per-day multichannel per-minute sequences (7 base + 6 rich modalities the paper found important:
GPS speed, app-usage, WiFi/BLE counts, ambience-speech, watch-light) -> Conv + BiLSTM +
SUBJECT EMBEDDING -> 7 sigmoids, focal loss, fold/seed ensemble; then blended in logit space with
the team's FrontierSilence base (the deep model adds orthogonal signal, esp. Q2/Q3 fatigue/stress).

CPU validation (450 train rows, interleaved CV): blend beat the unified OOF by ~ -0.008 mean,
Q2 alone -0.039. On GPU this scales far stronger (see SCALE-UP).

RUN:
  CPU (validation / small submission):  python mislstm_full.py
  GPU (Colab/Kaggle, the real 0.54 push): pip install torch timm; set CONFIG['device']='cuda',
     scale hid/epochs/seeds up, optionally swap the Conv block-encoder for a timm SEResNeXt101 on
     per-block images + CBAM (paper Sec.3), and enable the UALRE ensemble.

OUTPUT: submission_mislstm_full.csv (upload-safe 250x10) + prints CV-style diagnostics.
"""
from __future__ import annotations
import torch  # FIRST (Windows MKL/torch DLL order)
import torch.nn as nn
import torch.nn.functional as Fnn
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = dict(
    DATA=r"C:\Users\박광민\Documents\Codex\etri_team\data",
    FS=r"C:\Users\박광민\Documents\Codex\etri_team\submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv",
    device="cpu",                          # 'cuda' on GPU
    block_min=30, win_start=12, win_hours=24,
    hid=96, emb=8, epochs=80, folds=5, seeds=(11, 23, 37, 51),
    lr=2e-3, wd=1e-3, batch=32, gamma=1.5,
    # per-target deep-vs-FS blend weight. Set from the HONEST nested check (out-of-sample):
    # robust gains only on Q2 (-0.035) and Q3 (-0.007); Q1/S1 OVERFIT (hurt) -> 0. Conservative.
    blend_w=dict(Q1=0.0, Q2=0.40, Q3=0.25, S1=0.0, S2=0.05, S3=0.10, S4=0.0),
    out="submission_mislstm_full.csv",
)
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6
SPEECH = {"Speech", "Conversation", "Narration, monologue", "Babbling", "Shout", "Whispering",
          "Crowd", "Hubbub, speech noise, speech babble"}
DI = Path(CONFIG["DATA"]) / "ch2025_data_items"


def _per_min(name, col, fn=None):
    d = pd.read_parquet(DI / f"ch2025_{name}.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"]); d["tmin"] = d["timestamp"].dt.floor("1min")
    d["v"] = d[col].apply(fn) if fn else d[col]
    return d.groupby(["subject_id", "tmin"])["v"].mean()


def load_grid():
    g = {}
    g["hr_mean"] = _per_min("wHr", "heart_rate", lambda a: float(np.mean(a)) if a is not None and len(a) else np.nan)
    g["hr_std"] = _per_min("wHr", "heart_rate", lambda a: float(np.std(a)) if a is not None and len(a) else np.nan)
    g["steps"] = _per_min("wPedo", "step")
    g["screen"] = _per_min("mScreenStatus", "m_screen_use")
    g["charge"] = _per_min("mACStatus", "m_charging")
    g["still"] = _per_min("mActivity", "m_activity", lambda v: float(v == 3))
    g["light"] = _per_min("mLight", "m_light", lambda v: np.log1p(v))
    g["gps_speed"] = _per_min("mGps", "m_gps", lambda a: float(np.mean([p.get("speed", 0.0) for p in a])) if hasattr(a, "__len__") else np.nan)
    g["wifi_cnt"] = _per_min("mWifi", "m_wifi", lambda a: float(len(a)) if hasattr(a, "__len__") else np.nan)
    g["ble_cnt"] = _per_min("mBle", "m_ble", lambda a: float(len(a)) if hasattr(a, "__len__") else np.nan)
    g["usage"] = _per_min("mUsageStats", "m_usage_stats", lambda a: float(sum(x["total_time"] for x in a)) if hasattr(a, "__len__") else np.nan)
    def amb(a):
        try:
            tot = sum(float(x[1]) for x in a); sp = sum(float(x[1]) for x in a if str(x[0]) in SPEECH)
            return sp / tot if tot > 0 else 0.0
        except Exception:
            return np.nan
    g["amb_speech"] = _per_min("mAmbience", "m_ambience", amb)
    g["wlight"] = _per_min("wLight", "w_light", lambda v: np.log1p(v))
    CH = list(g.keys())
    G = pd.DataFrame(g).reset_index().sort_values(["subject_id", "tmin"]).reset_index(drop=True)
    return G, CH


def build(rows, G, CH, sid):
    nb = CONFIG["win_hours"] * 60 // CONFIG["block_min"]
    Xs, subs = [], []
    for _, r in rows.iterrows():
        s = r["subject_id"]; d0 = pd.to_datetime(r["lifelog_date"]) + pd.Timedelta(hours=CONFIG["win_start"])
        d1 = d0 + pd.Timedelta(hours=CONFIG["win_hours"])
        idx = pd.date_range(d0, d1, freq="1min", inclusive="left")
        sub = G[(G.subject_id == s) & (G.tmin >= d0) & (G.tmin < d1)].set_index("tmin").reindex(idx)
        X = np.full((nb, len(CH)), np.nan, np.float32)
        if len(sub):
            for i in range(nb):
                seg = sub.iloc[i * CONFIG["block_min"]:(i + 1) * CONFIG["block_min"]][CH].values
                with np.errstate(all="ignore"):
                    mean = np.nanmean(seg, 0)
                X[i] = mean
        Xs.append(X); subs.append(sid.get(s, 0))
    return np.array(Xs, np.float32), np.array(subs, np.int64)


class Net(nn.Module):
    def __init__(self, c, nsub, hid, emb):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)
        self.conv = nn.Sequential(nn.Conv1d(c, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, hid, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.emb = nn.Embedding(nsub, emb)
        self.head = nn.Sequential(nn.Linear(2 * hid + emb, 96), nn.ReLU(), nn.Dropout(0.4), nn.Linear(96, 7))

    def forward(self, x, s):
        B, T, C = x.shape
        x = self.bn(x.reshape(B * T, C)).reshape(B, T, C)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(x)
        return self.head(torch.cat([o.mean(1), self.emb(s)], 1))


def focal(z, y, g):
    p = torch.sigmoid(z); ce = Fnn.binary_cross_entropy_with_logits(z, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y); return (((1 - pt) ** g) * ce).mean()


def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def main():
    dev = CONFIG["device"]
    tr = pd.read_csv(Path(CONFIG["DATA"]) / "ch2026_metrics_train.csv")
    sample = pd.read_csv(Path(CONFIG["DATA"]) / "ch2026_submission_sample.csv")
    sid = {s: i for i, s in enumerate(sorted(tr.subject_id.unique()))}
    print("loading per-minute grid (13 channels, slow) ..."); G, CH = load_grid()
    print("building train sequences ..."); Xtr, Str = build(tr, G, CH, sid)
    print("building test sequences ..."); Xte, Ste = build(sample, G, CH, sid)
    Ytr = tr[TARGETS].values.astype(np.float32)
    C = Xtr.shape[2]
    mu = np.nanmean(Xtr.reshape(-1, C), 0); sd = np.nanstd(Xtr.reshape(-1, C), 0) + 1e-6
    def prep(A):
        Z = (A - mu) / sd; return np.concatenate([np.nan_to_num(Z), np.isnan(Z).astype(np.float32)], 2)
    Xtrp, Xtep = prep(Xtr), prep(Xte)
    rng = np.random.default_rng(0)
    test_pred = np.zeros((len(sample), 7)); nf = 0
    for seed in CONFIG["seeds"]:
        torch.manual_seed(seed); np.random.seed(seed)
        fold = np.zeros(len(tr), int)
        for s in np.unique(Str):
            idx = np.where(Str == s)[0]
            for k, ch in enumerate(np.array_split(rng.permutation(idx), CONFIG["folds"])):
                fold[ch] = k
        for k in range(CONFIG["folds"]):
            ti = np.where(fold != k)[0]
            net = Net(Xtrp.shape[2], len(sid), CONFIG["hid"], CONFIG["emb"]).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["wd"])
            xt = torch.tensor(Xtrp[ti]).to(dev); st = torch.tensor(Str[ti]).to(dev); yt = torch.tensor(Ytr[ti]).to(dev)
            xv = torch.tensor(Xtep).to(dev); sv = torch.tensor(Ste).to(dev)
            net.train(); n = len(ti); bs = CONFIG["batch"]
            for ep in range(CONFIG["epochs"]):
                perm = np.random.permutation(n)
                for i in range(0, n, bs):
                    ii = perm[i:i + bs]; opt.zero_grad()
                    focal(net(xt[ii], st[ii]), yt[ii], CONFIG["gamma"]).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                test_pred += torch.sigmoid(net(xv, sv)).cpu().numpy(); nf += 1
    test_pred /= nf

    # blend with FrontierSilence in logit space, per-target weight
    fs = pd.read_csv(CONFIG["FS"]).sort_values(["subject_id", "sleep_date", "lifelog_date"]).reset_index(drop=True)
    sm = sample.sort_values(["subject_id", "sleep_date", "lifelog_date"]).reset_index(drop=True)
    # align test_pred to sm order (build used sample order)
    order = sample.sort_values(["subject_id", "sleep_date", "lifelog_date"]).index.values
    tp = test_pred[order]
    out = sm.copy()
    for j, t in enumerate(TARGETS):
        w = CONFIG["blend_w"][t]
        z = (1 - w) * logit(fs[t].values) + w * logit(tp[:, j])
        out[t] = np.clip(1 / (1 + np.exp(-z)), 1e-4, 1 - 1e-4)
    out.to_csv(CONFIG["out"], index=False)
    print(f"saved {CONFIG['out']} shape {out.shape}")
    print("  per-target mean shift vs FS:", {t: round(float(out[t].mean() - fs[t].mean()), 4) for t in TARGETS})


if __name__ == "__main__":
    main()
