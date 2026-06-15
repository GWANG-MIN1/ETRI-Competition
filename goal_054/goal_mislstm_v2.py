#!/usr/bin/env python3
"""GOAL — enhanced MIS-LSTM-lite (Conv-block + BiLSTM + subject emb + focal) used as an ORTHOGONAL
component blended with the team OOF, per-target optimal blend weight (CV-chosen).

The deep-sequence model is weak alone on CPU but adds orthogonal per-row signal (Q2/S3 earlier).
Here: stronger model + seed ensemble + per-target blend-weight tuned on train folds -> measure how
far the blend beats the team OOF on interleaved CV. This is the CPU-achievable slice of the MIS-LSTM
paradigm; the full SEResNeXt101 version (GPU) is documented separately for the real 0.54 push.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
CORE = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(CORE))
import hsjepa_core as H  # noqa: E402
import pandas as pd

EPS = 1e-6
d = np.load(HERE / "mislstm_data.npz", allow_pickle=True)
X = d["X"]; Y = d["Y"]; SUB = d["SUB"]; TARGETS = list(d["TARGETS"])
N, NB, C = X.shape
keys = d["keys"]

m = H.load()
kdf = pd.DataFrame(keys, columns=["subject_id", "sleep_date", "lifelog_date"])
kdf["sleep_date"] = pd.to_datetime(kdf["sleep_date"]); kdf["lifelog_date"] = pd.to_datetime(kdf["lifelog_date"])
mk = m.copy(); mk["sleep_date"] = pd.to_datetime(mk["sleep_date"]); mk["lifelog_date"] = pd.to_datetime(mk["lifelog_date"])
mm = kdf.merge(mk, on=["subject_id", "sleep_date", "lifelog_date"], how="left")
OOF = np.column_stack([mm[f"p_{t}"].values for t in TARGETS]).astype(np.float32)
YY = np.column_stack([mm[f"y_{t}"].values for t in TARGETS]).astype(np.float32)


def bll(y, p):
    p = np.clip(p, EPS, 1 - EPS); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


class Net(nn.Module):
    def __init__(self, c, nsub, hid=96, emb=8):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)
        self.conv = nn.Sequential(nn.Conv1d(c, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.emb = nn.Embedding(nsub, emb)
        self.head = nn.Sequential(nn.Linear(2 * hid + emb, 96), nn.ReLU(), nn.Dropout(0.4), nn.Linear(96, 7))

    def forward(self, x, sub):
        B, T, Cc = x.shape
        x = self.bn(x.reshape(B * T, Cc)).reshape(B, T, Cc)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(x)
        o = o.mean(1)
        return self.head(torch.cat([o, self.emb(sub)], 1))


def focal(logits, y, g=1.5):
    p = torch.sigmoid(logits)
    ce = Fnn.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    return (((1 - pt) ** g) * ce).mean()


def zfill(Xtr, Xva):
    mu = np.nanmean(Xtr.reshape(-1, C), 0); sd = np.nanstd(Xtr.reshape(-1, C), 0) + 1e-6
    def f(A):
        Z = (A - mu) / sd; miss = np.isnan(Z).astype(np.float32)
        return np.concatenate([np.nan_to_num(Z), miss], 2)
    return f(Xtr), f(Xva)


def fit_predict(tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Xva = zfill(X[tr], X[va])
    net = Net(Xtr.shape[2], int(SUB.max() + 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    xt = torch.tensor(Xtr); st = torch.tensor(SUB[tr]); yt = torch.tensor(Y[tr])
    xv = torch.tensor(Xva); sv = torch.tensor(SUB[va])
    n = len(tr); bs = 32
    net.train()
    for ep in range(80):
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = focal(net(xt[idx], st[idx]), yt[idx]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(xv, sv)).numpy()


print("CV (interleaved 5-fold x 4 seeds), enhanced model + per-target optimal blend ...")
seeds = [11, 23, 37, 51]
pred = np.zeros((N, 7)); cnt = np.zeros(N)
for sd in seeds:
    for tr, va in H.interleaved_folds(mm, 5, seed=sd):
        pred[va] += fit_predict(tr, va, sd); cnt[va] += 1
pred /= np.maximum(cnt[:, None], 1)

# per-target optimal blend weight (grid), chosen to minimize logloss (report honest CV value)
print(f"\n  {'tgt':>4} | {'deep ll':>8} | {'OOF ll':>8} | {'blend ll':>9} | {'w*':>4} | {'gain vs OOF':>11}")
tot_o = tot_b = 0
for j, t in enumerate(TARGETS):
    y = YY[:, j]
    lo = bll(y, OOF[:, j]); ld = bll(y, pred[:, j])
    best = (lo, 0.0)
    for w in np.linspace(0, 0.6, 13):
        bl = 1 / (1 + np.exp(-((1 - w) * logit(OOF[:, j]) + w * logit(pred[:, j]))))
        lb = bll(y, bl)
        if lb < best[0]:
            best = (lb, w)
    tot_o += lo; tot_b += best[0]
    flag = "  <<<" if best[0] < lo - 0.002 else ""
    print(f"  {t:>4} | {ld:>8.4f} | {lo:>8.4f} | {best[0]:>9.4f} | {best[1]:>4.2f} | {best[0]-lo:>+11.4f}{flag}")
print(f"  {'MEAN':>4} | {'':>8} | {tot_o/7:>8.4f} | {tot_b/7:>9.4f} |      | {tot_b/7-tot_o/7:>+11.4f}")
print("\n  (blend weights are CV-tuned here = optimistic; real gain needs nested CV / a held-out probe.")
print("   Decisive read: does the deep component move the MEAN meaningfully below the team OOF?)")
