"""GOAL_064 — RED-TEAM "detectable but not correctable". Attack via 3 mechanisms.

Wedge: overshoot already IS a correctable direction (subject-level Q recency drift) -> the claim
is overstated. Test whether an ADDITIONAL correctable direction exists at:
  M1 aggregate(subject) level   : does subject train-mean residual predict FUTURE residual sign?
  M2 latent drift trajectory    : does HS-JEPA drift coord predict signed residual on FUTURE?
  M3 regime-conditional         : does a signed sensor-regime g make residual sign predictable
                                  (monotone), even if marginal mean ~0?
Each per target, FUTURE block, vs PLACEBO. Q2/Q3 positive = sanity (=overshoot). A positive on
S/Q1 BEYOND overshoot would BREAK the conclusion.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
sys.path.insert(0, r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa

WMC = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\worldmodel_jepa\cache")
TARGETS = list(K.TARGETS); SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173]
def hbin(h): hh = h if h >= 12 else h + 24; return int((hh - 12) * 12)
m = K.base(); subj = m["subject_id"].values
RES = {t: (m[f"y_{t}"].values - m[f"p_{t}"].values) for t in TARGETS}

def future_mask(seed, ff=0.15):
    rng = np.random.default_rng(seed); fut = np.zeros(len(m), bool)
    for s, idx in m.groupby("subject_id").groups.items():
        idx = np.array(sorted(idx, key=lambda i: m.at[i, "sleep_date"])); fut[idx[-max(1, int(round(len(idx) * ff))):]] = True
    return fut

# ---- M1: subject-level directional persistence ----
print("=== M1: subject train-mean residual -> FUTURE residual (aggregate direction) ===")
print("  tgt | corr(real) | placebo | future-Δlogloss(per-subj relevel) | break?")
def clip01(p): return np.clip(p, 1e-6, 1 - 1e-6)
def bll(y, p): p = clip01(p); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
m1 = {}
for t in TARGETS:
    r = RES[t]; y = m[f"y_{t}"].values; p = m[f"p_{t}"].values
    cr, cp, dl = [], [], []
    for seed in SEEDS:
        fut = future_mask(seed); tr = ~fut
        rbar = {s: float(r[tr & (subj == s)].mean()) if (tr & (subj == s)).any() else 0.0 for s in np.unique(subj)}
        pred = np.array([rbar[s] for s in subj])
        cr.append(pearsonr(pred[fut], r[fut])[0]) if np.std(pred[fut]) > 1e-9 else None
        rng = np.random.default_rng(seed); us = np.unique(subj); perm = rng.permutation(us)
        rmap = dict(zip(us, perm)); predp = np.array([rbar[rmap[s]] for s in subj])
        cp.append(pearsonr(predp[fut], r[fut])[0]) if np.std(predp[fut]) > 1e-9 else None
        # per-subject relevel by train rbar (in prob space, clipped): does it help future logloss?
        pc = clip01(p + pred); dl.append(bll(y[fut], pc[fut]) - bll(y[fut], p[fut]))
    cr = np.nanmean([x for x in cr if x is not None]); cp = np.nanmean([x for x in cp if x is not None]); dl = np.nanmean(dl)
    brk = (cr > 0.10 and cr - cp > 0.05 and dl < -0.001)
    m1[t] = (cr, cp, dl, brk)
    print(f"  {t}  | {cr:+.3f}     | {cp:+.3f}  | {dl:+.5f}                      | {'YES' if brk else 'no'}")

# ---- M2: HS-JEPA drift coordinate -> signed residual (FUTURE) ----
print("\n=== M2: HS-JEPA drift coord -> signed residual (FUTURE) ===")
df = np.load(WMC / "wm_drift_feature.npz", allow_pickle=True)
keys = df["keys"]; f700 = df["f"]
cmap = {(s, pd.to_datetime(dt).normalize()): i for i, (s, dt) in enumerate(keys)}
ridx = np.array([cmap[(s, pd.to_datetime(dd).normalize())] for s, dd in zip(m.subject_id, m.lifelog_date)])
drift = f700[ridx]
print("  tgt | corr(drift,resid) FUTURE | placebo | break?")
m2 = {}
for t in TARGETS:
    r = RES[t]; cr, cp = [], []
    for seed in SEEDS:
        fut = future_mask(seed)
        if np.std(drift[fut]) > 1e-9 and np.std(r[fut]) > 1e-9:
            cr.append(pearsonr(drift[fut], r[fut])[0])
        rng = np.random.default_rng(seed * 3); dp = drift[rng.permutation(len(drift))]
        cp.append(pearsonr(dp[fut], r[fut])[0])
    cr = np.nanmean(cr); cp = np.nanmean(cp)
    brk = abs(cr) > 0.10 and abs(cr) - abs(cp) > 0.05
    m2[t] = (cr, cp, brk)
    note = " (Q-overshoot sanity)" if t in ("Q2", "Q3") else (" <<< NEW" if brk else "")
    print(f"  {t}  | {cr:+.3f}                  | {cp:+.3f}  | {'YES' if brk else 'no'}{note}")

# ---- M3: regime-conditional direction (signed sensor deviation from recent baseline) ----
print("\n=== M3: signed sensor-regime g -> residual (marginal corr + high-|g| stratum), FUTURE ===")
d = np.load(WMC / "wm_canvas.npz", allow_pickle=True)
X = d["X"]; obs = d["obs"]; chans = list(d["chans"]); ci = {c: i for i, c in enumerate(chans)}
nb = slice(hbin(22), hbin(33))
def night_mean(ch):
    return np.array([X[r, ci[ch], nb][obs[r, ci[ch], nb] > 0].mean() if (obs[r, ci[ch], nb] > 0).any() else np.nan for r in ridx])
# signed regime: tonight's night value minus subject trailing(7-day) median, label-free
mdf = m.copy(); mdf["_d"] = pd.to_datetime(mdf["lifelog_date"])
def signed_recent_dev(ch):
    nm = night_mean(ch); g = np.full(len(m), np.nan)
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s); order = idx[np.argsort(mdf["_d"].values[idx])]
        for k, i in enumerate(order):
            prev = order[max(0, k - 7):k]
            if len(prev) >= 2 and np.isfinite(nm[i]):
                base = np.nanmedian(nm[prev]); sd = np.nanstd(nm[prev]) or 1.0
                g[i] = (nm[i] - base) / sd
    return g
regimes = {c: signed_recent_dev(c) for c in ("hr_mean", "active_rate", "screen_on") if c in ci}
print("  tgt | best signed-g | marginal corr | high|g| corr | placebo | break?")
m3 = {}
for t in TARGETS:
    r = RES[t]; best = None
    for gname, g in regimes.items():
        ok = np.isfinite(g)
        cm_, ch_, cpl = [], [], []
        for seed in SEEDS:
            fut = future_mask(seed) & ok
            if fut.sum() < 12 or np.std(g[fut]) < 1e-9:
                continue
            cm_.append(pearsonr(g[fut], r[fut])[0])
            hi = fut & (np.abs(g) >= np.nanpercentile(np.abs(g[ok]), 66))
            if hi.sum() >= 8 and np.std(g[hi]) > 1e-9:
                ch_.append(pearsonr(g[hi], r[hi])[0])
            rng = np.random.default_rng(seed * 11); gp = g[rng.permutation(len(g))]
            cpl.append(pearsonr(gp[fut], r[fut])[0])
        cm_m = np.nanmean(cm_) if cm_ else 0.0; ch_m = np.nanmean(ch_) if ch_ else 0.0; cpl_m = np.nanmean(cpl) if cpl else 0.0
        score = max(abs(cm_m), abs(ch_m))
        if best is None or score > best[0]:
            best = (score, gname, cm_m, ch_m, cpl_m)
    _, gn, cmg, chg, cpg = best
    brk = (abs(chg) > 0.12 and abs(chg) - abs(cpg) > 0.06) or (abs(cmg) > 0.10 and abs(cmg) - abs(cpg) > 0.05)
    m3[t] = (gn, cmg, chg, cpg, brk)
    print(f"  {t}  | {gn:12s} | {cmg:+.3f}        | {chg:+.3f}      | {cpg:+.3f}  | {'YES' if brk else 'no'}")

# ---- verdict ----
print("\n================ RED-TEAM VERDICT ================")
breaks = []
for t in TARGETS:
    if t in ("Q2", "Q3"):
        continue  # these are the known overshoot direction; only NEW (non-Q-recency) breaks count
    if m1[t][3]:
        breaks.append(f"M1:{t}")
    if m2[t][2]:
        breaks.append(f"M2:{t}")
    if m3[t][4]:
        breaks.append(f"M3:{t}")
print("  Q2/Q3 = known correctable direction (overshoot) -> conclusion ALREADY overstated.")
print(f"  NEW correctable-direction breaks (beyond Q-recency): {breaks if breaks else 'NONE'}")
if breaks:
    print("  -> conclusion BREAKS: an additional correctable direction exists; pursue + gate.")
else:
    print("  -> no ADDITIONAL correctable direction found; refined conclusion holds:")
    print("     'only correctable direction = Q-recency overshoot (already deployed); no new one'.")
