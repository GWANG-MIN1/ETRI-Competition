"""GOAL_060 Exp1 / Stage 2 — TRANSFER Walch-trained sleep model to ETRI, derive TST/SE/SOL/WASO
proxies, test vs S labels + transfer gate + placebo. Decisive S-side answer.

Train sleep/wake (LR, the saner Stage-1 model) on ALL Walch using ETRI-transferable feats
(HR-bpm stats + step activity, subject-normalized, 60s). Apply to each ETRI night
(lifelog_date 20:00 -> +1d 12:00) -> p_sleep/min -> per-night proxies. Then per S target:
AUC vs label, logit-blend onto base, authoritative transfer gate, shuffle placebo.
PASS iff a proxy beats base AND beats placebo AND gate PASS_LEVEL (esp. S3). Else GOAL_058/060 FAIL.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

KIT = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(HSJEPA_SRC))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa
from gate_transfer_vet import vet_candidate  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from walch_stage1_sanity import build_subject, subject_ids, FEATS  # reuse loader

EPS = 1e-6
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))

# ---- train sleep/wake on ALL Walch ----
subs = subject_ids(); D = {s: build_subject(s) for s in subs}; D = {s: d for s, d in D.items() if d}
Xtr = np.vstack([d[0] for d in D.values()]); ytr = np.concatenate([d[1] for d in D.values()])
scaler = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=1.0, max_iter=3000).fit(scaler.transform(Xtr), ytr)
print(f"Walch sleep/wake model trained on {len(ytr)} epochs ({len(D)} subj).")

# ---- ETRI per-minute features (must match Walch FEATS exactly) ----
m = K.base()
whr = K.modality("wHr"); wpedo = K.modality("wPedo")
whr["timestamp"] = pd.to_datetime(whr["timestamp"]); wpedo["timestamp"] = pd.to_datetime(wpedo["timestamp"])
# per-minute HR stats from the list column
def hr_stats(lst):
    a = np.asarray(lst, float); a = a[(a > 20) & (a < 240)]
    if len(a) == 0:
        return (np.nan,) * 4
    return (a.mean(), a.min(), a.max(), a.std() if len(a) > 1 else 0.0)
hh = whr.copy()
stats = np.array([hr_stats(x) for x in hh["heart_rate"].values])
hh["hr_mean"], hh["hr_min"], hh["hr_max"], hh["hr_std"] = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
hh = hh.dropna(subset=["hr_mean"])
# subject HR median/IQR (label-free, like Walch)
sub_med = hh.groupby("subject_id")["hr_mean"].median().to_dict()
sub_iqr = (hh.groupby("subject_id")["hr_mean"].quantile(0.75) - hh.groupby("subject_id")["hr_mean"].quantile(0.25)).to_dict()
hh["minute"] = hh["timestamp"].dt.floor("min")
hh = hh.set_index(["subject_id", "minute"])
ped = wpedo.copy(); ped["minute"] = ped["timestamp"].dt.floor("min")
ped_step = ped.groupby(["subject_id", "minute"])["step"].sum()

def night_proxies(sid, sleep_date):
    L = pd.to_datetime(sleep_date) - pd.Timedelta(days=1)
    start = L.normalize() + pd.Timedelta(hours=20); end = L.normalize() + pd.Timedelta(hours=36)  # +1d 12:00
    mins = pd.date_range(start, end, freq="min")
    try:
        sub_hh = hh.loc[sid]
    except KeyError:
        return None
    feats = []
    idx_ok = []
    iqr = sub_iqr.get(sid, 1.0) or 1.0; med = sub_med.get(sid, 70.0)
    for t in mins:
        if t not in sub_hh.index:
            continue
        r = sub_hh.loc[t]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        st = ped_step.get((sid, t), 0.0)
        feats.append([r["hr_mean"], r["hr_min"], r["hr_max"], r["hr_std"],
                      (r["hr_mean"] - med) / iqr, float(st), float(st > 0)])
        idx_ok.append(t)
    if len(feats) < 60:                      # need >=1h HR coverage in the night
        return None
    Xf = np.array(feats, float)
    ps = clf.predict_proba(scaler.transform(Xf))[:, 1]   # p_sleep per minute
    # proxies
    tst = float(ps.sum())                                 # est sleep minutes
    sl = ps > 0.5
    # onset = first index of >=10-min sustained sleep
    onset = None
    for i in range(len(sl) - 10):
        if sl[i:i + 10].mean() >= 0.8:
            onset = i; break
    if onset is None:
        return dict(tst=tst, se=np.nan, sol=np.nan, waso=np.nan, n=len(ps))
    last = len(sl) - 1
    while last > onset and not sl[last]:
        last -= 1
    sol = float(onset)                                    # minutes from window start to onset
    waso = float((~sl[onset:last + 1]).sum())             # wake minutes within sleep period
    se = float(sl[onset:last + 1].mean())                 # sleep efficiency in the sleep period
    return dict(tst=tst, se=se, sol=sol, waso=waso, n=len(ps))

prox = {k: np.full(len(m), np.nan) for k in ("tst", "se", "sol", "waso")}
cov = 0
for i, row in m.iterrows():
    r = night_proxies(row["subject_id"], row["sleep_date"])
    if r is None:
        continue
    cov += 1
    for k in prox:
        prox[k][i] = r.get(k, np.nan)
print(f"ETRI nights with proxy: {cov}/{len(m)}")

# ---- evaluate each proxy vs its S target (oriented AUC) + gate ----
PAIR = {"S1": "tst", "S2": "se", "S3": "sol", "S4": "waso"}
print("\n=== proxy AUC vs S label (coverage; oriented) + standalone ===")
cand = m[H.KEYS].copy()
for t in H.TARGETS:
    cand[t] = m[f"p_{t}"].values
for S, pk in PAIR.items():
    p = prox[pk]; ok = np.isfinite(p); y = m[f"y_{S}"].values
    if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
        print(f"  {S}<-{pk}: insufficient ({ok.sum()})"); continue
    auc = roc_auc_score(y[ok], p[ok]); auc = max(auc, 1 - auc)
    # blend proxy (subject-z -> logit nudge) onto base for the gate
    z = np.zeros(len(m)); pv = p[ok]
    z_ok = (pv - np.nanmean(pv)) / (np.nanstd(pv) + 1e-9)
    sign = 1.0 if roc_auc_score(y[ok], p[ok]) >= 0.5 else -1.0
    nudge = np.zeros(len(m)); nudge[ok] = sign * 0.3 * z_ok
    cand[S] = sig(logit(m[f"p_{S}"].values) + nudge)
    print(f"  {S}<-{pk}: AUC {auc:.3f}  (cov {ok.sum()}/{len(m)})  base_rate {y.mean():.3f}")

r = vet_candidate(cand, label="walch_transfer_Sproxies", verbose=False)
print("\n=== transfer gate (proxy-blended S vs base) ===")
for S in ("S1", "S2", "S3", "S4"):
    v = r["targets"][S]
    print(f"  {S}: verdict {v['verdict']:20s} d_full {v.get('d_full', float('nan')):+.4f} "
          f"resid {v.get('resid_gain', float('nan')):+.4f} placebo {v.get('placebo_gain', float('nan')):+.4f}")
pass_level = [S for S in ("S1", "S2", "S3", "S4") if r["targets"][S]["verdict"] == "PASS_LEVEL"]
print(f"\nPASS_LEVEL S targets: {pass_level if pass_level else 'NONE'}")
print(f"GOAL_060 Exp1 VERDICT: {'PASS' if pass_level else 'FAIL'} "
      f"-> {'external PSG transfer adds transferable S signal' if pass_level else 'no transferable S gain; S-side closed'}")
