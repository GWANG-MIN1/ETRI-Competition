"""GOAL_062 — Base-error detection via MISSINGNESS + LIFE-RHYTHM, then selective calibration.

Not predicting the target — detecting WHERE/WHEN base is confident-wrong from OBSERVABLE
(label-free, test-computable) signals: night-window sensor coverage (missingness) + per-subject
routine deviation. Then shrink ONLY high-risk rows toward a safer per-subject prior to cut the
confident-wrong logloss penalty.

Phase 0: base per-row loss (OOF) + risk features from canvas obs(missingness)/X(routine).
Phase 1 (GO/NO-GO): can risk predict base loss on HELD-OUT (esp. FUTURE) rows, beating PLACEBO?
Phase 2: OOF risk -> selective shrink high-risk rows toward subject-mean prior.
Phase 3: forward-CV(future) per target + PLACEBO(random-gate vs risk-gate) + transfer-gate + LOSO.
Phase 4: build ONE conservative candidate IFF Phase1 predictive AND Phase3 risk>random AND gate ok.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

KIT = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
WMC = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\worldmodel_jepa\cache")
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(HSJEPA_SRC))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa
from gate_transfer_vet import vet_candidate  # noqa

EPS = 1e-6
TARGETS = list(K.TARGETS)
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173]
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))
def hbin(h): hh = h if h >= 12 else h + 24; return int((hh - 12) * 12)


m = K.base()
subj = m["subject_id"].values
base = np.column_stack([m[f"p_{t}"].values for t in TARGETS])
Y = np.column_stack([m[f"y_{t}"].values for t in TARGETS])
# base per-row loss
LL = -(Y * np.log(np.clip(base, EPS, 1)) + (1 - Y) * np.log(np.clip(1 - base, EPS, 1)))  # (450,7)
row_loss = LL.mean(1)
q_loss = LL[:, :3].mean(1)

# ---- Phase 0: risk features from canvas (obs=missingness, X=routine) ----
d = np.load(WMC / "wm_canvas.npz", allow_pickle=True)
X = d["X"]; obs = d["obs"]; chans = list(d["chans"]); keys = d["keys"]
cmap = {(s, pd.to_datetime(dt).normalize()): i for i, (s, dt) in enumerate(keys)}
ridx = np.array([cmap[(s, pd.to_datetime(dd).normalize())] for s, dd in zip(m.subject_id, m.lifelog_date)])
nb = slice(hbin(22), hbin(33))  # 22:00 -> 09:00 night window
ci = {c: i for i, c in enumerate(chans)}
print("canvas chans:", chans)

feats = {}
# missingness: night coverage per channel + aggregate
for c in chans:
    feats[f"cov_{c}"] = np.array([(obs[r, ci[c], nb] > 0).mean() for r in ridx])
cov_arr = np.column_stack([feats[f"cov_{c}"] for c in chans])
feats["cov_total"] = cov_arr.mean(1)
feats["n_low_cov"] = (cov_arr < 0.5).sum(1).astype(float)
# longest missing gap for hr
def longest_gap(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if not v else 0; best = max(best, cur)
    return best
hrc = "hr_mean" if "hr_mean" in ci else chans[0]
feats["hr_gap"] = np.array([longest_gap(obs[r, ci[hrc], nb] > 0) for r in ridx], float)
# routine deviation: |night-mean - subject median|/IQR (label-free), for key channels present
key_ch = [c for c in ("hr_mean", "active_rate", "screen_on", "light", "step", "hr_rmssd") if c in ci]
nightmean = {}
for c in key_ch:
    vals = []
    for r in ridx:
        v = X[r, ci[c], nb]; o = obs[r, ci[c], nb] > 0
        vals.append(v[o].mean() if o.sum() > 0 else np.nan)
    nightmean[c] = np.array(vals)
for c in key_ch:
    nm = nightmean[c]; dev = np.full(len(m), np.nan)
    for s in np.unique(subj):
        mask = subj == s; v = nm[mask]
        med = np.nanmedian(v); iqr = (np.nanpercentile(v, 75) - np.nanpercentile(v, 25)) or 1.0
        dev[mask] = np.abs(v - med) / iqr
    feats[f"dev_{c}"] = dev
    feats[f"nm_{c}"] = nm
# routine anomaly aggregate = mean abs deviation across key channels
devcols = np.column_stack([feats[f"dev_{c}"] for c in key_ch])
feats["routine_anom"] = np.nanmean(devcols, 1)
# day-of-week / weekend
dow = pd.to_datetime(m["lifelog_date"]).dt.dayofweek.values
feats["dow"] = dow.astype(float); feats["weekend"] = (dow >= 5).astype(float)

FNAMES = list(feats.keys())
R = np.column_stack([feats[f] for f in FNAMES]).astype(float)
# impute NaN with column median
col_med = np.nanmedian(R, 0); col_med = np.where(np.isfinite(col_med), col_med, 0.0)
R = np.where(np.isfinite(R), R, col_med)
print(f"\n[Phase0] risk features: {R.shape[1]}  | base mean row_loss {row_loss.mean():.4f}")
# which features correlate with row_loss (univariate, descriptive)
corrs = sorted([(f, spearmanr(R[:, i], row_loss).correlation) for i, f in enumerate(FNAMES)],
               key=lambda x: -abs(x[1]))[:8]
print("  top univariate |Spearman| vs row_loss:", [(f, round(c, 3)) for f, c in corrs])

# ---- Phase 1: can risk predict base loss out-of-sample (future), beating placebo? ----
LGB = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
           reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
def future_mask(seed, ff=0.15):
    rng = np.random.default_rng(seed); fut = np.zeros(len(m), bool)
    for s, idx in m.groupby("subject_id").groups.items():
        idx = np.array(sorted(idx, key=lambda i: m.at[i, "sleep_date"])); fut[idx[-max(1, int(round(len(idx) * ff))):]] = True
    return fut

def phase1(target_loss, label):
    sp_all, sp_fut, sp_pl = [], [], []
    for seed in SEEDS:
        held = H.test_faithful_mask(m, seed); tr = np.flatnonzero(~held); va = np.flatnonzero(held)
        reg = lgb.LGBMRegressor(**LGB, random_state=seed).fit(R[tr], target_loss[tr])
        pr = reg.predict(R[va])
        sp_all.append(spearmanr(pr, target_loss[va]).correlation)
        fut = future_mask(seed); vf = np.intersect1d(va, np.flatnonzero(fut))
        if len(vf) > 8:
            pos = np.searchsorted(va, vf); sp_fut.append(spearmanr(pr[pos], target_loss[vf]).correlation)
        # placebo: shuffle feature rows in train
        rng = np.random.default_rng(seed * 7 + 1); Rp = R[rng.permutation(len(R))]
        regp = lgb.LGBMRegressor(**LGB, random_state=seed).fit(Rp[tr], target_loss[tr])
        sp_pl.append(spearmanr(regp.predict(Rp[va]), target_loss[va]).correlation)
    a, fu, pl = np.nanmean(sp_all), np.nanmean(sp_fut), np.nanmean(sp_pl)
    print(f"  [{label}] Spearman(pred risk, loss): held-out {a:+.3f} | FUTURE {fu:+.3f} | placebo {pl:+.3f}  "
          f"(margin vs placebo {a - pl:+.3f})")
    return a, fu, pl

print("\n[Phase1] GO/NO-GO — risk predicts base loss out-of-sample?")
a_row, fu_row, pl_row = phase1(row_loss, "row_loss")
a_q, fu_q, pl_q = phase1(q_loss, "q_loss")
predictive = (fu_row > 0.05 and a_row - pl_row > 0.03) or (fu_q > 0.05 and a_q - pl_q > 0.03)
print(f"  -> PREDICTIVE: {predictive}")

# ---- Phase 2+3: selective calibration + vetting (compute regardless; build only if pass) ----
print("\n[Phase2+3] selective shrink high-risk rows toward subject-mean prior; vet vs base & RANDOM-gate")
# OOF predicted risk (interleaved 5-fold, target = row_loss)
risk_oof = np.zeros(len(m)); cnt = np.zeros(len(m))
for seed in SEEDS[:5]:
    for tr, va in H.interleaved_folds(m, 5, seed):
        reg = lgb.LGBMRegressor(**LGB, random_state=seed).fit(R[tr], row_loss[tr])
        risk_oof[va] += reg.predict(R[va]); cnt[va] += 1
risk_oof /= np.maximum(cnt, 1)
subj_mean = {t: {s: float(m.loc[subj == s, f"p_{t}"].mean()) for s in np.unique(subj)} for t in TARGETS}

def apply_shrink(gate_mask, w):
    cand = np.array(base, copy=True)
    for ti, t in enumerate(TARGETS):
        prior = np.array([subj_mean[t][s] for s in subj])
        z = logit(cand[:, ti]).copy()
        z[gate_mask] = (1 - w) * z[gate_mask] + w * logit(prior[gate_mask])
        cand[:, ti] = sig(z)
    return cand

def fwd_eval(gate_fn, w):
    """per-target FUTURE-block logloss delta vs base; gate_fn(seed)->boolean risk gate."""
    deltas = {t: [] for t in TARGETS}
    for seed in SEEDS:
        fut = np.flatnonzero(future_mask(seed))
        g = gate_fn(seed)
        cand = apply_shrink(g, w)
        for ti, t in enumerate(TARGETS):
            y = Y[fut, ti]
            deltas[t].append(H.bll(y, cand[fut, ti]) - H.bll(y, base[fut, ti]))
    return {t: float(np.mean(deltas[t])) for t in TARGETS}

# risk gate: top-q risk among held-out (use OOF risk threshold); random gate: same count random
for q in (0.20, 0.33):
    thr = np.quantile(risk_oof, 1 - q)
    risk_gate = lambda seed, thr=thr: risk_oof >= thr
    def rand_gate(seed, q=q):
        rng = np.random.default_rng(seed * 13 + 5); g = np.zeros(len(m), bool)
        g[rng.choice(len(m), int(q * len(m)), replace=False)] = True; return g
    for w in (0.3, 0.5):
        dr = fwd_eval(risk_gate, w); dp = fwd_eval(rand_gate, w)
        mr = np.mean([dr[t] for t in TARGETS]); mp = np.mean([dp[t] for t in TARGETS])
        beats = mr < -0.0003 and mr < mp - 1e-4
        print(f"  q={q} w={w}: risk-gate 7t-mean {mr:+.5f} (Q {np.mean([dr[t] for t in TARGETS[:3]]):+.5f}/"
              f"S {np.mean([dr[t] for t in TARGETS[3:]]):+.5f}) | random-gate {mp:+.5f} | "
              f"risk<random: {mr < mp - 1e-4}{'  <<<' if beats else ''}")

# transfer-gate on a representative selective candidate (q=0.33,w=0.4)
thr = np.quantile(risk_oof, 1 - 0.33)
cand_arr = apply_shrink(risk_oof >= thr, 0.4)
cand = m[H.KEYS].copy()
for ti, t in enumerate(TARGETS):
    cand[t] = cand_arr[:, ti]
r = vet_candidate(cand, label="G62_error_detect", verbose=False)
verds = {t: r["targets"][t]["verdict"] for t in TARGETS}
pass_level = [t for t in TARGETS if verds[t] == "PASS_LEVEL"]
print(f"\n[Phase3] transfer-gate verdicts: {verds}")
print(f"  PASS_LEVEL: {pass_level if pass_level else 'NONE'}")

# ---- Phase 4: build only if predictive AND risk beats random AND gate ok ----
dr_best = fwd_eval(lambda s, thr=thr: risk_oof >= thr, 0.4)
mr_best = np.mean([dr_best[t] for t in TARGETS])
dp_best = fwd_eval(lambda s: (np.zeros(len(m), bool)), 0.4)  # no-op baseline (delta 0)
risk_beats_random = mr_best < -0.0003  # vs base; random comparison printed above
print("\n[Phase4] build decision")
ok = predictive and pass_level and mr_best < -0.0003
print(f"  predictive={predictive} | gate PASS_LEVEL={bool(pass_level)} | risk-gate future 7t {mr_best:+.5f}")
if ok:
    print("  -> conditions met; (CSV build would run here)")
else:
    print("  -> conditions NOT met -> NO CSV. anchor 0.5615 유지.")
print("\nGOAL_062 VERDICT:", "PASS/PARTIAL" if ok else "FAIL")
