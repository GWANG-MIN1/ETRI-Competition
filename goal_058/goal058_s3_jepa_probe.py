"""GOAL_058 option-2 — single lightweight probe: does the FROZEN HS-JEPA latent carry a
transferable S3 signal that beats placebo? (No full MIL build, no S2/S4, ~minutes.)

frozen HS-JEPA encoder output z[96] per night (cached) -> small supervised S3 HEAD
(logistic linear-probe + small MLP) -> logit-blend onto base. Compared against:
  - time_shuffle placebo : encoder run on a TIME-SHUFFLED night canvas (cache z_shuffle)
  - noise placebo        : encoder run on NOISE (cache z_noise)
  - rowperm placebo      : real z with rows permuted within subject (destroys z<->label pairing)
PASS (per user): real_z S3 beats base on test_faithful AND future block AND clearly beats
placebos AND the authoritative transfer gate is not NO_GAIN/LOTTERY. Else GOAL_058 = FAIL.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FHC = HERE.parent
KIT = FHC / "outputs"
WMC = FHC / "worldmodel_jepa" / "cache"
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(HSJEPA_SRC))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa
from gate_transfer_vet import vet_candidate  # noqa

EPS = 1e-6
SEEDS = [11, 23, 37, 51, 67, 83]
T = "S3"


def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))


m = H.load()
y = m[f"y_{T}"].values
base = m[f"p_{T}"].values
basez = logit(base)

# ---- align frozen z (and placebos) to the 450 train rows ----
zc = np.load(WMC / "wm_circajepa_z.npz", allow_pickle=True)
pl = np.load(WMC / "wm_placebo_z.npz", allow_pickle=True)
keys700 = np.load(WMC.parent / "cache" / "wm_drift_feature.npz", allow_pickle=True)["keys"]
cmap = {(s, pd.to_datetime(d).normalize()): i for i, (s, d) in enumerate(keys700)}
ridx = np.array([cmap[(s, pd.to_datetime(d).normalize())] for s, d in zip(m.subject_id, m.lifelog_date)])
Zr = zc["z"][ridx]
Zt = pl["z_shuffle"][ridx]
Zn = pl["z_noise"][ridx]
rng = np.random.default_rng(0); Zp = Zr.copy()
for s in m["subject_id"].unique():
    ii = np.flatnonzero(m["subject_id"].values == s); Zp[ii] = Zp[rng.permutation(ii)]
REPS = {"real_z": Zr, "time_shuffle": Zt, "noise": Zn, "rowperm": Zp}
print(f"aligned z: real {Zr.shape}  | S3 base test_faithful = "
      f"{np.mean([H.bll(y[H.test_faithful_mask(m, s)], base[H.test_faithful_mask(m, s)]) for s in SEEDS]):.5f}")


def split_future_inter(seed, ff=0.15, inf=0.30):
    rng = np.random.default_rng(seed); fut = np.zeros(len(m), bool); inter = np.zeros(len(m), bool)
    for s, idx in m.groupby("subject_id").groups.items():
        idx = np.array(sorted(idx, key=lambda i: m.at[i, "sleep_date"])); n = len(idx)
        nf = max(1, int(round(n * ff))); fut[idx[-nf:]] = True
        rest = idx[:-nf]; ni = max(1, int(round(n * inf)))
        inter[rng.choice(rest, size=min(ni, len(rest)), replace=False)] = True
    return np.flatnonzero(fut), np.flatnonzero(inter)


WS = [0.1, 0.2, 0.3, 0.5]


def train_head(Z, tr, va, seed, kind):
    sc = StandardScaler().fit(Z[tr]); Ztr, Zva = sc.transform(Z[tr]), sc.transform(Z[va])
    if kind == "logit":
        clf = LogisticRegression(C=0.2, max_iter=3000).fit(Ztr, y[tr])
    else:
        clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=3.0, max_iter=600,
                            early_stopping=True, random_state=seed).fit(Ztr, y[tr])
    return clf.predict_proba(Zva)[:, 1]


def evaluate(kind):
    print(f"\n=== HEAD = {kind} ===")
    rows = {}
    for name, Z in REPS.items():
        d_tf, d_fut = {w: [] for w in WS}, {w: [] for w in WS}
        for seed in SEEDS:
            held = H.test_faithful_mask(m, seed); tr = np.flatnonzero(~held); va = np.flatnonzero(held)
            ph = train_head(Z, tr, va, seed, kind)
            bb = base[va]; bll0 = H.bll(y[va], bb)
            fidx, _ = split_future_inter(seed)
            futva = np.intersect1d(va, fidx)
            for w in WS:
                pb = sig((1 - w) * logit(bb) + w * logit(ph))
                d_tf[w].append(H.bll(y[va], pb) - bll0)
                if len(futva):
                    pos = np.searchsorted(va, futva)
                    d_fut[w].append(H.bll(y[futva], pb[pos]) - H.bll(y[futva], base[futva]))
        bw = min(WS, key=lambda w: np.mean(d_tf[w]))
        tf = np.array(d_tf[bw]); fut = np.array(d_fut[bw])
        rows[name] = (bw, tf.mean(), float(np.mean(tf < 0)), fut.mean(), float(np.mean(fut < 0)))
    print(f"  {'rep':12s} | bestW | tf_Δ(sign)        | future_Δ(sign)")
    for name, (bw, tf, tfs, fu, fus) in rows.items():
        flag = " <<<" if name == "real_z" else ""
        print(f"  {name:12s} |  {bw:.1f}  | {tf:+.5f}({tfs:.2f}) | {fu:+.5f}({fus:.2f}){flag}")
    return rows


res_log = evaluate("logit")
res_mlp = evaluate("mlp")

# ---- authoritative transfer gate on the real_z logistic S3 candidate (interleaved OOF) ----
print("\n=== AUTHORITATIVE TRANSFER GATE (real_z logit, S3 candidate vs base) ===")
oof = np.full(len(m), np.nan); cnt = np.zeros(len(m))
for seed in SEEDS:
    for tr, va in H.interleaved_folds(m, 5, seed):
        ph = train_head(Zr, tr, va, seed, "logit")
        oof[va] = np.where(np.isnan(oof[va]), 0.0, oof[va]) + ph; cnt[va] += 1
oof = np.where(cnt > 0, oof / np.maximum(cnt, 1), base)
w = 0.3
s3blend = sig((1 - w) * basez + w * logit(oof))
cand = m[H.KEYS].copy()
for t in H.TARGETS:
    cand[t] = m[f"p_{t}"].values
cand["S3"] = s3blend
r = vet_candidate(cand, label=f"frozenJEPA_S3_blend_w{w}", verbose=True)

# ---- verdict ----
bw, tf, tfs, fu, fus = res_log["real_z"]
_, ptf, _, pfu, _ = res_log["time_shuffle"]
_, ntf, _, nfu, _ = res_log["rowperm"]
real_beats_base = tf < -0.0005 and fu < 0
real_beats_plac = (tf < ptf - 0.0005) and (tf < ntf - 0.0005)
gate_ok = r["targets"]["S3"]["verdict"] in ("PASS_LEVEL",)
print("\n" + "=" * 70)
print(f"S3 real_z(logit): tf_Δ {tf:+.5f}  future_Δ {fu:+.5f} | time_shuffle tf {ptf:+.5f} | rowperm tf {ntf:+.5f}")
print(f"  beats base: {real_beats_base} | clearly beats placebo: {real_beats_plac} | gate PASS_LEVEL: {gate_ok}")
verdict = "PASS" if (real_beats_base and real_beats_plac and gate_ok) else "FAIL"
print(f"\nGOAL_058 PROBE VERDICT (S3) = {verdict}")
if verdict == "FAIL":
    print("  -> frozen HS-JEPA latent shows no transferable, placebo-beating S3 signal.")
    print("  -> GOAL_058 = FAIL; anchor 0.5615333471 유지; S-target reconstruction 방향 중단.")
print("=" * 70)
