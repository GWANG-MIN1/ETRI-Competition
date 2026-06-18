"""sp_07: PRE-SUBMIT AUDIT for the LAST submission ticket.

NO new candidates, NO new exploration. Sole purpose: verify that C1
(submission_twobucket_int0p6_fut1p0_317adcf1_uploadsafe.csv) is a safe FINAL upload vs the
anchor best (submission_overshoot_x0p8_330ef1a1_uploadsafe.csv, LB 0.5615333471).

Checks: (1) file exists; (2) exact format match to sample submission; (3) anchor-vs-C1 diff
(only Q2/Q3 changed, Q1/S1-4 byte-identical, interleaved overshoot 0.8->0.6 = closer to base,
future overshoot 0.8->1.0 = further from base, deltas == (sigma-0.8)*step exactly); (4) re-print
PE / forward-CV / worst-loose for C1 and why C1 > C2 for a single safe ticket; (5) VERDICT A or B.
"""
from __future__ import annotations
import sys
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
C1_F = ETRI / "submission_twobucket_int0p6_fut1p0_317adcf1_uploadsafe.csv"
C2_F = ETRI / "submission_twobucket_int0p4_fut1p2_50deaad1_uploadsafe.csv"
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173, 197, 211]


def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))


PASS = []  # (ok, label)
def chk(ok, label, detail=""):
    PASS.append(bool(ok))
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


print("=" * 78)
print("PRE-SUBMIT AUDIT  —  last ticket  —  C1 (int0.6,fut1.0) vs anchor 0.5615")
print("=" * 78)

# ----------------------------------------------------------------- 1. EXISTENCE
print("\n[1] FILE EXISTENCE")
chk(C1_F.exists(), f"C1 exists: {C1_F.name}", f"({C1_F.stat().st_size} bytes)" if C1_F.exists() else "MISSING")
chk(ANCHOR_F.exists(), f"anchor exists: {ANCHOR_F.name}")
assert C1_F.exists() and ANCHOR_F.exists(), "missing file — cannot audit"

sample = pd.read_csv(ETRI / "data" / "ch2026_submission_sample.csv")
c1 = pd.read_csv(C1_F)
anchor = pd.read_csv(ANCHOR_F)

# ------------------------------------------------------ 2. FORMAT vs SAMPLE
print("\n[2] FORMAT — exact match to ch2026_submission_sample.csv")
chk(c1.shape == (250, 10), "shape == (250,10)", f"got {c1.shape}")
chk(list(c1.columns) == list(sample.columns), "column names+order identical",
    f"{list(c1.columns)}")
chk(len(c1) == 250, "row count == 250", f"got {len(c1)}")
chk([c for c in TARGETS if c in c1.columns] == TARGETS, "7 target columns present", str(TARGETS))
# id / order identical to sample (the exact upload order)
order_ok = c1[KEYS].reset_index(drop=True).equals(sample[KEYS].reset_index(drop=True))
chk(order_ok, "KEYS rows + order identical to sample (id/order)")
prob = c1[TARGETS].to_numpy(float)
chk(int(c1[TARGETS].isnull().sum().sum()) == 0, "no NaN in targets")
chk(bool(prob.min() >= 0.0 and prob.max() <= 1.0), "probabilities in [0,1]",
    f"min {prob.min():.6f} max {prob.max():.6f}")
chk(bool(prob.min() > 0.0 and prob.max() < 1.0), "strictly in (0,1) (logloss-safe)",
    f"min {prob.min():.6f} max {prob.max():.6f}")

# --------------------------------------------- 3. ANCHOR vs C1 DIFF
print("\n[3] ANCHOR vs C1 — what changed")
mg = anchor[KEYS + TARGETS].merge(c1[KEYS + TARGETS], on=KEYS, suffixes=("_a", "_c"))
chk(len(mg) == 250, "anchor & C1 share all 250 keys", f"matched {len(mg)}")
changed = {}
for t in TARGETS:
    d = np.abs(mg[f"{t}_a"].to_numpy(float) - mg[f"{t}_c"].to_numpy(float))
    changed[t] = (d > 1e-12).sum()
chk([t for t in TARGETS if changed[t] > 0] == ["Q2", "Q3"], "ONLY Q2/Q3 changed",
    " ".join(f"{t}:{changed[t]}" for t in TARGETS))
for t in ("Q1", "S1", "S2", "S3", "S4"):
    md = np.abs(mg[f"{t}_a"].to_numpy(float) - mg[f"{t}_c"].to_numpy(float)).max()
    chk(md == 0.0, f"{t} byte-identical to anchor", f"max|Δ|={md:.2e}")

# bucket classification on the FILE rows (sample order)
fsub = c1["subject_id"].values
fdates = pd.to_datetime(c1["sleep_date"]).values
train = pd.read_csv(ETRI / "data" / "ch2026_metrics_train.csv")
tmax = {s: np.datetime64(v) for s, v in pd.to_datetime(train.groupby("subject_id")["sleep_date"].max()).items()}
is_future = np.array([fdates[i] > tmax[fsub[i]] for i in range(len(c1))])
chk(abs(is_future.mean() - 0.376) < 0.02, "bucket split ~37.6% future / 62.4% interleaved",
    f"future={int(is_future.sum())} inter={int((~is_future).sum())}")

# direction check: need FS base + per-subject d_over step
print("\n    direction (σ_int 0.8->0.6 = closer to base; σ_fut 0.8->1.0 = further from base):")
PE.init()
S = PE._S; subs = S["subs"]; vix = S["vix"]; d_over = S["d_over"]
fs = pd.read_csv(ETRI / PE.FS_FILE)
fsb = fs.set_index(KEYS).loc[c1.set_index(KEYS).index]  # FS base aligned to C1 order
for t in ("Q2", "Q3"):
    step = np.array([d_over[vix[(s, t)]] for s in fsub])
    base_z = logit(fsb[t].to_numpy(float))
    a_z = logit(mg[f"{t}_a"].to_numpy(float))
    c_z = logit(mg[f"{t}_c"].to_numpy(float))
    nz = np.abs(step) > 1e-9  # rows with actual drift step
    # exact delta vs expected (sigma - 0.8) * step
    exp = np.where(is_future, (1.0 - 0.8), (0.6 - 0.8)) * step
    max_err = np.abs((c_z - a_z) - exp).max()
    chk(max_err < 1e-9, f"{t}: C1−anchor logit == (σ_row−0.8)·step exactly", f"max err {max_err:.2e}")
    # interleaved closer to base, future further from base (where step≠0)
    im = nz & ~is_future; fm = nz & is_future
    int_closer = np.all(np.abs(c_z[im] - base_z[im]) < np.abs(a_z[im] - base_z[im]))
    fut_further = np.all(np.abs(c_z[fm] - base_z[fm]) > np.abs(a_z[fm] - base_z[fm]))
    ri = np.median(np.abs(c_z[im] - base_z[im]) / np.abs(a_z[im] - base_z[im])) if im.sum() else float('nan')
    rf = np.median(np.abs(c_z[fm] - base_z[fm]) / np.abs(a_z[fm] - base_z[fm])) if fm.sum() else float('nan')
    chk(int_closer, f"{t}: interleaved overshoot REDUCED (closer to base)", f"ratio≈{ri:.3f} (expect 0.75) n={int(im.sum())}")
    chk(fut_further, f"{t}: future overshoot INCREASED (further from base)", f"ratio≈{rf:.3f} (expect 1.25) n={int(fm.sum())}")

# integrity: on-disk C1 == intended build(0.6,1.0)
def build(si, sf):
    out = fs.sort_values(KEYS).reset_index(drop=True)[KEYS].copy()
    fsS = fs.sort_values(KEYS).reset_index(drop=True)
    bsub = fsS["subject_id"].values
    bdates = pd.to_datetime(fsS["sleep_date"]).values
    bfut = np.array([bdates[i] > tmax[bsub[i]] for i in range(len(fsS))])
    for t in TARGETS:
        z = logit(fsS[t].to_numpy(float)).copy()
        if t in ("Q2", "Q3"):
            stp = np.array([d_over[vix[(s, t)]] for s in bsub])
            z = z + np.where(bfut, sf, si) * stp
        out[t] = sig(z)
    return out.set_index(KEYS).loc[sample.set_index(KEYS).index].reset_index()
rebuilt = build(0.6, 1.0)
integ = max(np.abs(rebuilt[t].to_numpy(float) - c1[t].to_numpy(float)).max() for t in TARGETS)
chk(integ < 1e-12, "on-disk C1 == intended build(σ_int=0.6, σ_fut=1.0)", f"max|Δ|={integ:.2e}")

# ------------------------------------------------- 4. ARBITER SUMMARY
print("\n[4] ARBITER SUMMARY (re-output)")
def pe_avg(si, sf):
    delta = {}
    for s in subs:
        m = fsub == s
        avg = (is_future[m].sum() * sf + (~is_future[m]).sum() * si) / m.sum()
        for t in ("Q2", "Q3"):
            delta[(s, t)] = avg * d_over[vix[(s, t)]]
    return PE.eval_delta(delta, settings18=True, verbose=False)
dep = pe_avg(0.8, 0.8); c1pe = pe_avg(0.6, 1.0); c2pe = pe_avg(0.4, 1.2)
print(f"  deployed best (0.8,0.8): PE post-mean {dep['post_mean']:+.5f}  worst-loose {dep['worst_loose']:+.5f}")
print(f"  C1 (0.6,1.0)           : PE post-mean {c1pe['post_mean']:+.5f}  worst-loose {c1pe['worst_loose']:+.5f}"
      f"  fav18 {c1pe['fav18']:.2f}")
print(f"     -> PE Δ vs best: {c1pe['post_mean'] - dep['post_mean']:+.5f}   "
      f"worst-loose {c1pe['worst_loose']:+.5f} ({'SAFE/negative' if c1pe['worst_loose'] < 0 else 'POSITIVE=risk'})")
print(f"  C2 (0.4,1.2)           : PE post-mean {c2pe['post_mean']:+.5f}  worst-loose {c2pe['worst_loose']:+.5f}"
      f"  ({'POSITIVE=risk' if c2pe['worst_loose'] > 0 else 'safe'})")

# forward-CV for C1
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
M = K.base(); fc = {}
for t in ("Q2", "Q3"):
    g = []
    for sd in SEEDS:
        fi, ii = split_fi(M, sd); tr = np.setdiff1d(np.arange(len(M)), np.concatenate([fi, ii]))
        stp = rstep(M, tr, t); subj = M["subject_id"].values; y = M[f"y_{t}"].values; z0 = logit(M[f"p_{t}"].values)
        s = np.array([stp.get(x, 0.0) for x in subj])
        p08 = sig(z0 + 0.8 * s); ll0 = 0.376 * H.bll(y[fi], p08[fi]) + 0.624 * H.bll(y[ii], p08[ii])
        pz = z0.copy(); pz[fi] = z0[fi] + 1.0 * s[fi]; pz[ii] = z0[ii] + 0.6 * s[ii]
        ll1 = 0.376 * H.bll(y[fi], sig(pz[fi])) + 0.624 * H.bll(y[ii], sig(pz[ii]))
        g.append(ll1 - ll0)
    fc[t] = (float(np.mean(g)), float(np.mean(np.array(g) < 0)))
seven = (fc["Q2"][0] + fc["Q3"][0]) / 7.0
print(f"  forward-CV (vs uniform-0.8): Q2 {fc['Q2'][0]:+.5f}(sign {fc['Q2'][1]:.2f})  "
      f"Q3 {fc['Q3'][0]:+.5f}(sign {fc['Q3'][1]:.2f})  | 7-target Δ {seven:+.6f}")
pred_lb = 0.5615333471 + (c1pe['post_mean'] - dep['post_mean'])
print(f"  predicted LB (PE-calibrated): 0.5615333471 {c1pe['post_mean'] - dep['post_mean']:+.5f} -> ~{pred_lb:.5f}")

# --------------------------------------------------------- VERDICT
print("\n" + "=" * 78)
all_ok = all(PASS)
c1_safe = c1pe['worst_loose'] < 0
print(f"format/diff checks: {sum(PASS)}/{len(PASS)} passed | C1 worst-loose {'negative (safe)' if c1_safe else 'POSITIVE'}")
if all_ok and c1_safe:
    print("CONCLUSION A:")
    print(f"FINAL_SUBMIT = {C1_F.name}")
    print("이 파일을 마지막 제출로 사용해도 됩니다.")
else:
    print("CONCLUSION B:")
    print("파일 형식/검증 문제 발견. 마지막 제출은 anchor best를 유지해야 합니다.")
    print(f"  FINAL_SUBMIT = {ANCHOR_F.name}")
print("=" * 78)
