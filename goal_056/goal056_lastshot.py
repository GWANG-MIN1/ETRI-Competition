"""GOAL_056 — LAST-SHOT aggressive rank-rescue: TARGET-SPECIFIC Q2/Q3 two-bucket.

NO new model, NO leakage. Same logit-space recency-overshoot formulation as the anchor
(FS base + sigma * per-subject Q2/Q3 drift step), but with FOUR independent sigmas:
  sigma_Q2_interleaved, sigma_Q2_future, sigma_Q3_interleaved, sigma_Q3_future
bucket = (row sleep_date > subject's max train date)  [observed; transferable level class].
Only Q2/Q3 change; Q1/S1-4 stay byte-identical to the anchor.

Grid: interleaved in {0.0,0.2,0.4,0.6,0.8}, future in {0.8,1.0,1.2,1.4,1.6} per target
=> 25 x 25 = 625 joint configs. PE post-mean is EXACTLY additive across targets (proved: gain
is a sum of per-(subject,target) terms), so each target's 25 configs are swept once and summed.
worst-loose (joint, NOT additive) is computed via the loose polytope for a shortlist.

Selection (user rules): hard gate worst-loose <= +0.0015; TOP priority PE Δ <= -0.0015 &
worst-loose <= +0.0015; else PE Δ <= -0.0010 & worst-loose <= +0.0015; else compare C1/C2.
Outputs a ranking table + builds EXACTLY ONE final CSV.
"""
from __future__ import annotations
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FHC = HERE.parent
KIT = FHC / "outputs"
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(FHC)); sys.path.insert(0, str(KIT)); sys.path.insert(0, str(HSJEPA_SRC))

import polytope_eval as PE  # noqa: E402
import goal054_kit as K  # noqa: E402
import hsjepa_core as H  # noqa: E402

EPS = 1e-6
KEYS = PE.KEYS
TARGETS = PE.TARGETS
ETRI = PE.ETRI
ANCHOR_F = ETRI / "submission_overshoot_x0p8_330ef1a1_uploadsafe.csv"
ANCHOR_LB = 0.5615333471
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173, 197, 211]
INTS = [0.0, 0.2, 0.4, 0.6, 0.8]
FUTS = [0.8, 1.0, 1.2, 1.4, 1.6]


def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))


print("init PE ...")
PE.init()
S_ = PE._S
subs = S_["subs"]; vix = S_["vix"]; d_over = S_["d_over"]; Smat = S_["S"]; n_var = S_["n_var"]

fs = pd.read_csv(ETRI / PE.FS_FILE).sort_values(KEYS).reset_index(drop=True)
sample = pd.read_csv(ETRI / "data" / "ch2026_submission_sample.csv").sort_values(KEYS).reset_index(drop=True)
anchor = pd.read_csv(ANCHOR_F)
fsub = fs["subject_id"].values
fdates = pd.to_datetime(fs["sleep_date"]).values
train = pd.read_csv(ETRI / "data" / "ch2026_metrics_train.csv")
tmax = {s: np.datetime64(v) for s, v in pd.to_datetime(train.groupby("subject_id")["sleep_date"].max()).items()}
is_future = np.array([fdates[i] > tmax[fsub[i]] for i in range(len(fs))])
print(f"buckets: future={int(is_future.sum())} ({is_future.mean():.3f}) inter={int((~is_future).sum())}")

# per-subject future/inter counts (for per-subject average sigma = PE granularity)
cnt = {s: (int(is_future[fsub == s].sum()), int((~is_future[fsub == s]).sum())) for s in subs}


# -------------------------------------------------- PE post-mean (additive, fast)
def pm_t(t, si, sf):
    """post-mean contribution of target t's two-bucket (per-subject avg sigma)."""
    delta = np.zeros(n_var)
    for s in subs:
        nf, ni = cnt[s]
        avg = (nf * sf + ni * si) / (nf + ni)
        delta[vix[(s, t)]] = avg * d_over[vix[(s, t)]]
    A, coefs = PE._gain_terms(delta)
    return float((Smat @ coefs + A).mean())


# sanity: additive pm vs full eval_delta on the anchor (0.8,0.8 both targets)
pm_best = pm_t("Q2", 0.8, 0.8) + pm_t("Q3", 0.8, 0.8)
ref_best = PE.eval_delta({(s, t): 0.8 * d_over[vix[(s, t)]] for s in subs for t in ("Q2", "Q3")}, verbose=False)
assert abs(pm_best - ref_best["post_mean"]) < 1e-9, f"additivity check FAILED {pm_best} vs {ref_best['post_mean']}"
print(f"additivity check OK: pm_best {pm_best:+.6f} == eval_delta {ref_best['post_mean']:+.6f}")
PM_REF = {"Q2": pm_t("Q2", 0.8, 0.8), "Q3": pm_t("Q3", 0.8, 0.8)}


# -------------------------------------------------- forward-CV (precompute per seed)
M = K.base()


def split_fi(m, seed, ff=0.15, inf=0.25):
    rng = np.random.default_rng(seed); fut = np.zeros(len(m), bool); inter = np.zeros(len(m), bool)
    for s, idx in m.groupby("subject_id").groups.items():
        idx = np.array(sorted(idx, key=lambda i: m.at[i, "sleep_date"])); n = len(idx)
        nf = max(1, int(round(n * ff))); f = idx[-nf:]; rest = idx[:-nf]
        ni = max(1, int(round(n * inf))); it = rng.choice(rest, size=min(ni, len(rest)), replace=False)
        fut[f] = True; inter[it] = True
    return np.flatnonzero(fut), np.flatnonzero(inter)


def rstep(m, tr, t, tau=21.0):
    trdf = m.iloc[tr]; st = {}
    for s, idx in trdf.groupby("subject_id").groups.items():
        d = trdf.loc[idx]; last = d["sleep_date"].max()
        w = np.exp(-(last - d["sleep_date"]).dt.days.values / tau)
        st[s] = logit(np.clip(float(np.sum(w * d[f"y_{t}"].values) / np.sum(w)), EPS, 1 - EPS)) - \
            logit(np.clip(float(d[f"y_{t}"].values.mean()), EPS, 1 - EPS))
    return st


pre = {"Q2": [], "Q3": []}
for t in ("Q2", "Q3"):
    for sd in SEEDS:
        fi, ii = split_fi(M, sd); tr = np.setdiff1d(np.arange(len(M)), np.concatenate([fi, ii]))
        step = rstep(M, tr, t); subj = M["subject_id"].values
        s = np.array([step.get(x, 0.0) for x in subj])
        pre[t].append((fi, ii, s, logit(M[f"p_{t}"].values), M[f"y_{t}"].values))


def fcv_t(t, si, sf):
    g = []
    for (fi, ii, s, z0, y) in pre[t]:
        p08 = sig(z0 + 0.8 * s); ll0 = 0.376 * H.bll(y[fi], p08[fi]) + 0.624 * H.bll(y[ii], p08[ii])
        pz = z0.copy(); pz[fi] = z0[fi] + sf * s[fi]; pz[ii] = z0[ii] + si * s[ii]
        ll1 = 0.376 * H.bll(y[fi], sig(pz[fi])) + 0.624 * H.bll(y[ii], sig(pz[ii]))
        g.append(ll1 - ll0)
    a = np.array(g); return float(a.mean()), float(np.mean(a < 0))


# per-target tables (25 configs each)
TBL = {"Q2": {}, "Q3": {}}
for t in ("Q2", "Q3"):
    for si in INTS:
        for sf in FUTS:
            dpm = pm_t(t, si, sf) - PM_REF[t]      # PE post-mean Δ vs anchor for this target
            fg, fsgn = fcv_t(t, si, sf)            # forward-CV gain vs uniform-0.8 for this target
            TBL[t][(si, sf)] = (dpm, fg, fsgn)


# -------------------------------------------------- joint table (625) + named
def joint(cfg):
    q2si, q2sf, q3si, q3sf = cfg
    dQ2, fQ2, sQ2 = TBL["Q2"][(q2si, q2sf)]
    dQ3, fQ3, sQ3 = TBL["Q3"][(q3si, q3sf)]
    pe_d = dQ2 + dQ3
    fcv7 = (fQ2 + fQ3) / 7.0
    ext = sum(1 for x in (q2sf, q3sf) if x >= 1.4) + sum(1 for x in (q2si, q3si) if x == 0.0)
    return dict(cfg=cfg, pe_d=pe_d, fcv7=fcv7, sQ2=sQ2, sQ3=sQ3, ext=ext)


grid = [joint((a, b, c, d)) for a in INTS for b in FUTS for c in INTS for d in FUTS]
named = {
    "C1": (0.6, 1.0, 0.6, 1.0), "C2": (0.4, 1.2, 0.4, 1.2),
    "corner1": (0.2, 1.4, 0.2, 1.4), "corner2": (0.0, 1.4, 0.2, 1.2),
    "corner3": (0.2, 1.6, 0.4, 1.2), "corner4": (0.4, 1.4, 0.2, 1.6),
}

# shortlist for worst-loose: top-60 by PE Δ + all named
grid_sorted = sorted(grid, key=lambda r: r["pe_d"])
short_cfgs = {r["cfg"] for r in grid_sorted[:60]} | set(named.values())


def full_eval(cfg):
    q2si, q2sf, q3si, q3sf = cfg
    delta = {}
    for s in subs:
        nf, ni = cnt[s]
        aQ2 = (nf * q2sf + ni * q2si) / (nf + ni)
        aQ3 = (nf * q3sf + ni * q3si) / (nf + ni)
        delta[(s, "Q2")] = aQ2 * d_over[vix[(s, "Q2")]]
        delta[(s, "Q3")] = aQ3 * d_over[vix[(s, "Q3")]]
    return PE.eval_delta(delta, settings18=True, verbose=False)


# build test predictions for a cfg (for L2 / mean-delta diagnostics + final save)
def build(cfg):
    q2si, q2sf, q3si, q3sf = cfg
    out = fs[KEYS].copy()
    smap = {"Q2": (q2si, q2sf), "Q3": (q3si, q3sf)}
    for t in TARGETS:
        z = logit(fs[t].values).copy()
        if t in ("Q2", "Q3"):
            si, sf = smap[t]
            z = z + np.where(is_future, sf, si) * np.array([d_over[vix[(s, t)]] for s in fsub])
        out[t] = sig(z)
    return out.set_index(KEYS).loc[sample.set_index(KEYS).index].reset_index()


anchor_al = anchor.set_index(KEYS).loc[sample.set_index(KEYS).index].reset_index()


def diagnostics(cfg):
    o = build(cfg)
    l2 = float(np.sqrt(sum(((o[t].to_numpy(float) - anchor_al[t].to_numpy(float)) ** 2).sum() for t in ("Q2", "Q3"))))
    md = {}
    for t in ("Q2", "Q3"):
        dz = logit(o[t].to_numpy(float)) - logit(anchor_al[t].to_numpy(float))
        md[t] = float(dz.mean())
        md[f"{t}_int"] = float(dz[~is_future].mean()); md[f"{t}_fut"] = float(dz[is_future].mean())
    return l2, md


rows = []
for cfg in short_cfgs:
    r = next(x for x in grid if x["cfg"] == cfg)
    fe = full_eval(cfg)
    l2, md = diagnostics(cfg)
    nm = next((k for k, v in named.items() if v == cfg), "")
    rows.append(dict(cfg=cfg, name=nm, pe_d=r["pe_d"], worst=fe["worst_loose"], fav=fe["fav18"],
                     fcv7=r["fcv7"], ext=r["ext"], l2=l2, md=md,
                     pred_lb=ANCHOR_LB + r["pe_d"]))

# ---------------------------------------------------------------- ranking output
rows.sort(key=lambda r: r["pe_d"])
print("\n=== SHORTLIST RANKING (sorted by PE Δ vs best; gate worst-loose<=+0.0015) ===")
print("rank cfg(Q2i,Q2f,Q3i,Q3f) name     PE_Δ      worst-loose  fcv7      ext  L2     predLB")
for i, r in enumerate(rows[:30]):
    c = r["cfg"]
    print(f"{i + 1:>3} ({c[0]:.1f},{c[1]:.1f},{c[2]:.1f},{c[3]:.1f}) {r['name']:<8} "
          f"{r['pe_d']:+.5f}  {r['worst']:+.5f}   {r['fcv7']:+.5f}  {r['ext']:>2}  {r['l2']:.3f}  {r['pred_lb']:.5f}")

# named baselines explicit
print("\n=== NAMED BASELINES ===")
for k in ("C1", "C2", "corner1", "corner2", "corner3", "corner4"):
    r = next(x for x in rows if x["name"] == k)
    c = r["cfg"]; md = r["md"]
    print(f"  {k:<8} ({c[0]:.1f},{c[1]:.1f},{c[2]:.1f},{c[3]:.1f})  PE_Δ {r['pe_d']:+.5f}  worst {r['worst']:+.5f}"
          f"  fcv7 {r['fcv7']:+.5f}  predLB {r['pred_lb']:.5f}")
    print(f"            meanΔlogit Q2 {md['Q2']:+.4f}(int {md['Q2_int']:+.4f}/fut {md['Q2_fut']:+.4f}) "
          f"Q3 {md['Q3']:+.4f}(int {md['Q3_int']:+.4f}/fut {md['Q3_fut']:+.4f})")

# ---------------------------------------------------------------- selection
GATE = 0.0015
safe = [r for r in rows if r["worst"] <= GATE]
tier1 = sorted([r for r in safe if r["pe_d"] <= -0.0015], key=lambda r: (r["pe_d"] + 0.0003 * r["ext"]))
tier2 = sorted([r for r in safe if r["pe_d"] <= -0.0010], key=lambda r: (r["pe_d"] + 0.0003 * r["ext"]))
print("\n=== SELECTION ===")
print(f"  worst-loose<=+{GATE}: {len(safe)} configs | tier1(PE_Δ<=-0.0015): {len(tier1)} | tier2(PE_Δ<=-0.0010): {len(tier2)}")

c1r = next(x for x in rows if x["name"] == "C1"); c2r = next(x for x in rows if x["name"] == "C2")
if tier1:
    pick = tier1[0]; reason = "tier1: PE_Δ<=-0.0015 AND worst-loose<=+0.0015 (top priority, ext-penalized)"
elif tier2:
    pick = tier2[0]; reason = "tier2: PE_Δ<=-0.0010 AND worst-loose<=+0.0015 (ext-penalized)"
else:
    pick = c2r if c2r["pe_d"] < c1r["pe_d"] and c2r["worst"] <= GATE else c1r
    reason = "no tier1/2 candidate; fell back to better of C1/C2"

# must be clearly better than C1 and C2 if it's a new candidate
if pick["name"] not in ("C1", "C2"):
    if not (pick["pe_d"] < c1r["pe_d"] - 1e-6 and pick["pe_d"] < c2r["pe_d"] - 1e-6):
        pick = c2r if c2r["worst"] <= GATE else c1r
        reason += " | reverted: new cand not clearly better than both C1 & C2"

c = pick["cfg"]; md = pick["md"]
print(f"\n  PICK: cfg ({c[0]:.1f},{c[1]:.1f},{c[2]:.1f},{c[3]:.1f}) name={pick['name'] or 'grid'}  | {reason}")
print(f"  PE_Δ {pick['pe_d']:+.5f}  worst-loose {pick['worst']:+.5f}  fav18 {pick['fav']:.2f}  "
      f"fcv7 {pick['fcv7']:+.5f}  L2 {pick['l2']:.3f}  predLB {pick['pred_lb']:.5f}")

# ---------------------------------------------------------------- build ONE csv
out = build(c)[sample.columns.tolist()]
prob = out[TARGETS].to_numpy(float)
dd = np.abs(prob - anchor_al[TARGETS].to_numpy(float))
checks = {
    "shape (250,10)": out.shape == (250, 10),
    "keys==sample": out[KEYS].reset_index(drop=True).equals(sample[KEYS].reset_index(drop=True)),
    "no NaN": int(out[TARGETS].isnull().sum().sum()) == 0,
    "strictly (0,1)": bool(prob.min() > 0 and prob.max() < 1),
    "only Q2/Q3 changed vs anchor": [t for i, t in enumerate(TARGETS) if dd[:, i].max() > 1e-12] == ["Q2", "Q3"],
}
assert all(checks.values()), f"upload-safety FAILED: {checks}"
digest = hashlib.sha1(np.round(prob, 12).tobytes()).hexdigest()[:8]
tag = f"Q2_{c[0]:.1f}_{c[1]:.1f}_Q3_{c[2]:.1f}_{c[3]:.1f}".replace(".", "p")
FN = ETRI / f"submission_lastshot_targetspec_{tag}_{digest}_uploadsafe.csv"
out.to_csv(FN, index=False)
print(f"\nupload-safety: ALL OK")
print(f"FINAL CSV: {FN.name}")
print(f"  hash {digest}  changed Q2/Q3 cells {int((dd > 1e-12).sum())}  L2(Q2/Q3) {pick['l2']:.3f}")
print(f"  meanΔlogit Q2 {md['Q2']:+.4f}(int {md['Q2_int']:+.4f}/fut {md['Q2_fut']:+.4f})  "
      f"Q3 {md['Q3']:+.4f}(int {md['Q3_int']:+.4f}/fut {md['Q3_fut']:+.4f})")
print("\nDONE")
