#!/usr/bin/env python3
"""GOAL — MIS-LSTM-lite training + CV vs team OOF (CPU). Validate the deep-sequence paradigm.

Model: per-block feature norm -> BiLSTM over 48 blocks -> concat SUBJECT EMBEDDING -> 7 sigmoids.
CV: interleaved 5-fold x seeds. Compare per-target logloss to the team unified OOF, and test a
blend (orthogonality). If MIS-LSTM-lite (or blend) beats OOF on CV, the deep-sequence + subject-
embedding paradigm carries real per-row signal the team's CatBoost/tiny-seqcnn missed -> build full
(GPU) version. If it ties/loses, scale (SEResNeXt101) is essential and CPU can't validate it.
"""
from __future__ import annotations
# import torch FIRST (Windows: importing numpy/MKL before torch breaks its DLL load)
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
CORE = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(CORE))
import hsjepa_core as H  # noqa: E402

torch.manual_seed(0)
EPS = 1e-6
d = np.load(HERE / "mislstm_data.npz", allow_pickle=True)
X = d["X"]; Y = d["Y"]; SUB = d["SUB"]; TARGETS = list(d["TARGETS"]); CH = list(d["CH"])
N, NB, C = X.shape
print(f"data X{X.shape} Y{Y.shape} subjects {len(np.unique(SUB))}")

# build missing mask, fill NaN with 0 (after per-channel z-norm computed on observed)
keys = d["keys"]  # [N,3] subject_id, sleep_date, lifelog_date


def bll(y, p):
    p = np.clip(p, EPS, 1 - EPS); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


# align team OOF to these rows
m = H.load()
import pandas as pd
kdf = pd.DataFrame(keys, columns=["subject_id", "sleep_date", "lifelog_date"])
kdf["sleep_date"] = pd.to_datetime(kdf["sleep_date"]); kdf["lifelog_date"] = pd.to_datetime(kdf["lifelog_date"])
mk = m.copy(); mk["sleep_date"] = pd.to_datetime(mk["sleep_date"]); mk["lifelog_date"] = pd.to_datetime(mk["lifelog_date"])
kdf["row"] = np.arange(len(kdf))
mm = kdf.merge(mk, on=["subject_id", "sleep_date", "lifelog_date"], how="left")
OOF = np.column_stack([mm[f"p_{t}"].values for t in TARGETS])  # [N,7]
YY = np.column_stack([mm[f"y_{t}"].values for t in TARGETS])


class Net(nn.Module):
    def __init__(self, c, nsub, hid=64, emb=8):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)
        self.lstm = nn.LSTM(c, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.emb = nn.Embedding(nsub, emb)
        self.head = nn.Sequential(nn.Linear(2 * hid + emb, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 7))

    def forward(self, x, sub):
        # x [B,T,C] -> bn over C
        B, T, C = x.shape
        x = self.bn(x.reshape(B * T, C)).reshape(B, T, C)
        o, _ = self.lstm(x)
        o = o.mean(1)
        e = self.emb(sub)
        return self.head(torch.cat([o, e], 1))


def zfill(Xtr, Xva):
    # per-channel mean/std over observed train cells
    mu = np.nanmean(Xtr.reshape(-1, C), 0); sd = np.nanstd(Xtr.reshape(-1, C), 0) + 1e-6
    def f(A):
        Z = (A - mu) / sd
        miss = np.isnan(Z).astype(np.float32)
        Z = np.nan_to_num(Z, nan=0.0)
        return np.concatenate([Z, miss], axis=2)   # add missing-mask channels
    return f(Xtr), f(Xva)


def train_eval(tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Xva = zfill(X[tr], X[va])
    Cc = Xtr.shape[2]
    net = Net(Cc, int(SUB.max() + 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.tensor(Xtr); st = torch.tensor(SUB[tr]); yt = torch.tensor(Y[tr])
    xv = torch.tensor(Xva); sv = torch.tensor(SUB[va])
    net.train()
    n = len(tr); bs = 32
    for ep in range(60):
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = net(xt[idx], st[idx])
            loss = lossf(out, yt[idx])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pv = torch.sigmoid(net(xv, sv)).numpy()
    return pv


print("\nCV (interleaved 5-fold x 3 seeds) — MIS-LSTM-lite vs team OOF per target ...")
seeds = [11, 23, 37]
pred = np.zeros((N, 7)); cnt = np.zeros(N)
for sd in seeds:
    for tr, va in H.interleaved_folds(mm, 5, seed=sd):
        pv = train_eval(tr, va, sd)
        pred[va] += pv; cnt[va] += 1
pred /= np.maximum(cnt[:, None], 1)

print(f"\n  {'tgt':>4} | {'MIS-LSTM ll':>11} | {'team OOF ll':>11} | {'blend ll':>9} | {'best vs OOF':>11}")
tot_m = tot_o = tot_b = 0
for j, t in enumerate(TARGETS):
    y = YY[:, j]
    lm = bll(y, pred[:, j])
    lo = bll(y, OOF[:, j])
    bl = 1 / (1 + np.exp(-(0.5 * logit(pred[:, j]) + 0.5 * logit(OOF[:, j]))))
    lb = bll(y, bl)
    tot_m += lm; tot_o += lo; tot_b += min(lb, lo)
    best = min(lm, lb) - lo
    flag = "  <<<" if best < -0.003 else ""
    print(f"  {t:>4} | {lm:>11.4f} | {lo:>11.4f} | {lb:>9.4f} | {best:>+11.4f}{flag}")
print(f"  {'MEAN':>4} | {tot_m/7:>11.4f} | {tot_o/7:>11.4f} | {tot_b/7:>9.4f} |")
print("\n  MIS-LSTM-lite or blend << OOF => deep-sequence+subject-embedding paradigm validated (build full on GPU).")
print("  ties/loses => CPU-scale insufficient; the SEResNeXt101-scale model is what top teams need.")
