#!/usr/bin/env python3
"""Quick CPU OOF check of the ANTI-OVERFIT config (HID64, dropout0.55, conv64, WD3e-3) on cached
13-ch data. Confirms the GPU overfit (Q2 OOF 0.71) is fixed before the user re-runs on Colab.
4 seeds for speed."""
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
CORE = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(CORE))
import hsjepa_core as H  # noqa
import pandas as pd
EPS = 1e-6
d = np.load(HERE / "mislstm_data_v3.npz", allow_pickle=True)
X = d["X"]; Y = d["Y"]; SUB = d["SUB"]; TARGETS = list(d["TARGETS"]); keys = d["keys"]
N, NB, C = X.shape
m = H.load(); kdf = pd.DataFrame(keys, columns=["subject_id", "sleep_date", "lifelog_date"])
kdf["sleep_date"] = pd.to_datetime(kdf["sleep_date"]); kdf["lifelog_date"] = pd.to_datetime(kdf["lifelog_date"])
mk = m.copy(); mk["sleep_date"] = pd.to_datetime(mk["sleep_date"]); mk["lifelog_date"] = pd.to_datetime(mk["lifelog_date"])
mm = kdf.merge(mk, on=["subject_id", "sleep_date", "lifelog_date"], how="left")
YY = np.column_stack([mm[f"y_{t}"].values for t in TARGETS]).astype(np.float32)
OOFu = np.column_stack([mm[f"p_{t}"].values for t in TARGETS]).astype(np.float32)


def bll(y, p):
    p = np.clip(p, EPS, 1 - EPS); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Net(nn.Module):
    def __init__(self, c, nsub, hid=64, emb=8):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)
        self.conv = nn.Sequential(nn.Conv1d(c, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, hid, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.emb = nn.Embedding(nsub, emb)
        self.head = nn.Sequential(nn.Linear(2 * hid + emb, 96), nn.ReLU(), nn.Dropout(0.55), nn.Linear(96, 7))

    def forward(self, x, s):
        B, T, Cc = x.shape
        x = self.bn(x.reshape(B * T, Cc)).reshape(B, T, Cc)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(x)
        return self.head(torch.cat([o.mean(1), self.emb(s)], 1))


def focal(z, y, g=1.5):
    p = torch.sigmoid(z); ce = Fnn.binary_cross_entropy_with_logits(z, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y); return (((1 - pt) ** g) * ce).mean()


def zf(Xtr, Xva):
    mu = np.nanmean(Xtr.reshape(-1, C), 0); sd = np.nanstd(Xtr.reshape(-1, C), 0) + 1e-6
    f = lambda A: np.concatenate([np.nan_to_num((A - mu) / sd), np.isnan((A - mu) / sd).astype(np.float32)], 2)
    return f(Xtr), f(Xva)


pred = np.zeros((N, 7)); cnt = np.zeros(N)
for sd in [11, 23, 37, 51]:
    torch.manual_seed(sd); np.random.seed(sd)
    for tr, va in H.interleaved_folds(mm, 5, seed=sd):
        Xtr, Xva = zf(X[tr], X[va])
        net = Net(Xtr.shape[2], int(SUB.max() + 1))
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=3e-3)
        xt = torch.tensor(Xtr); st = torch.tensor(SUB[tr]); yt = torch.tensor(Y[tr])
        net.train()
        for ep in range(70):
            perm = np.random.permutation(len(tr))
            for i in range(0, len(tr), 32):
                ii = perm[i:i + 32]; opt.zero_grad(); focal(net(xt[ii], st[ii]), yt[ii]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred[va] += torch.sigmoid(net(torch.tensor(Xva), torch.tensor(SUB[va]))).numpy(); cnt[va] += 1
pred /= np.maximum(cnt[:, None], 1)
print("ANTI-OVERFIT config OOF (HID64, dropout0.55, WD3e-3, 4 seeds):")
print(f"  {'tgt':>4} | {'deep OOF':>8} | {'unified OOF':>11} | {'global floor':>12}")
floor = {"Q1": .693, "Q2": .685, "Q3": .673, "S1": .625, "S2": .647, "S3": .640, "S4": .686}
for j, t in enumerate(TARGETS):
    print(f"  {t:>4} | {bll(YY[:, j], pred[:, j]):>8.4f} | {bll(YY[:, j], OOFu[:, j]):>11.4f} | {floor[t]:>12.3f}")
print("  (deep OOF should now be BELOW the global floor again, esp. Q2 ~0.64-0.65 not 0.71)")
