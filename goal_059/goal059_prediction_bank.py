"""GOAL_059 — PREDICTION-BANK ENSEMBLE RESET.

NO new model. Gather every existing OOF prediction (108-model zoo + named alternatives),
keep only directions DIFFERENT from the anchor (overshoot x0.8) AND not in a failed family
(Q2/Q3 overshoot, two-bucket, Sleep-JEPA / S-reconstruction, seqcnn/mislstm), then build 3
target-wise logit ensembles (conservative / balanced / aggressive-diversity) on top of the
anchor backbone. For each: changed targets + mean logit Δ vs anchor, test_faithful CV delta,
prediction distance, failed-direction overlap, and the authoritative transfer-gate verdict.

CSV is written ONLY when a candidate is (a) sufficiently DIFFERENT from anchor, (b) CV not
badly broken, (c) NOT overlapping a failed direction, AND (d) the transfer gate shows a real
PASS_LEVEL transferable gain (not LOTTERY/NO_GAIN — the documented CV-LB-gap trap).
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
ROOT = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690")
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(HSJEPA_SRC))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa
from gate_transfer_vet import vet_candidate  # noqa

EPS = 1e-6
TARGETS = list(K.TARGETS)


def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))


m = K.base()
base = np.column_stack([m[f"p_{t}"].values for t in TARGETS])          # unified base OOF
anchor = K.fs_proxy_arr(m, overshoot=0.8)                              # anchor-equivalent (overshoot x0.8) OOF
sc_base = K.score_perrow(m, base); sc_anch = K.score_perrow(m, anchor)
print(f"base   CV mean {K.meanll(sc_base):.5f} | anchor(overshoot0.8) CV mean {K.meanll(sc_anch):.5f}")
print(f"  (anchor worse than base on honest CV by {K.meanll(sc_anch)-K.meanll(sc_base):+.5f} — its LB win is the "
      f"CV-invisible global overshoot)")

# ---------------- build the bank ----------------
FAILED_PAT = ("seqcnn", "mislstm")  # failed paradigm OOFs (memory); recon/twobucket/sleepjepa are test-side
zoo = sorted((ROOT / "outputs" / "models").glob("**/oof.csv"))
named = [KIT / f for f in ["knn_oof_K25_cross.csv", "knn_oof_K25_within.csv", "rocket_oof.csv",
                           "seqcnn_oof_train.csv", "goal054_mislstm_oof_small_s0.csv",
                           "goal054_per_target_zoo_select_OOF.csv", "decorrelated_zoo_blend_oof_t06_equal.csv"]]
bank = []
for p in zoo + named:
    if not p.exists():
        continue
    try:
        arr = K.align(m, K.load_oof_csv(p))
    except Exception:
        continue
    if np.isnan(arr).all():
        continue
    nm = "/".join(p.parts[-3:]) if "models" in str(p) else p.name
    fam = "failed" if any(f in nm.lower() for f in FAILED_PAT) else (
        "knn" if "knn" in nm.lower() else "rocket" if "rocket" in nm.lower() else
        "meta" if ("zoo_select" in nm.lower() or "decorrelated" in nm.lower()) else "zoo")
    bank.append(dict(name=nm, arr=arr, fam=fam))
print(f"bank loaded: {len(bank)} members "
      f"(zoo {sum(b['fam']=='zoo' for b in bank)}, knn {sum(b['fam']=='knn' for b in bank)}, "
      f"rocket {sum(b['fam']=='rocket' for b in bank)}, meta {sum(b['fam']=='meta' for b in bank)}, "
      f"failed {sum(b['fam']=='failed' for b in bank)})")

# ---------------- per-member, per-target: CV + decorrelation vs anchor direction ----------------
anchor_dir = logit(np.clip(anchor, EPS, 1 - EPS)) - logit(np.clip(base, EPS, 1 - EPS))  # (450,7) anchor's move vs base
for b in bank:
    a = np.where(np.isnan(b["arr"]), base, b["arr"])
    b["arr"] = a
    b["resid"] = logit(a) - logit(base)                  # member's direction vs base
    b["cv"] = K.score_perrow(m, a)
    cor = {}
    for ti, t in enumerate(TARGETS):
        d = b["resid"][:, ti]; ad = anchor_dir[:, ti]
        cor[t] = float(np.corrcoef(d, ad)[0, 1]) if d.std() > 1e-9 and ad.std() > 1e-9 else 0.0
    b["corr_anchor"] = cor

# diversity pool per target: decorrelated from anchor, NOT failed family, target-CV not much worse than base
def pool_for(t, ti):
    out = []
    for b in bank:
        if b["fam"] == "failed":
            continue
        if abs(b["corr_anchor"][t]) > 0.5:           # too aligned with anchor's existing move
            continue
        if b["cv"][t] > sc_base[t] + 0.02:           # standalone target CV not garbage
            continue
        out.append(b)
    out.sort(key=lambda b: b["cv"][t])               # best standalone CV first
    return out[:6]


# ---------------- build 3 ensembles (anchor backbone + decorrelated residuals) ----------------
def build_ens(total_w):
    """ens_logit_t = logit(anchor_t) + (total_w distributed over decorrelated pool residuals_t)."""
    ze = logit(np.clip(anchor, EPS, 1 - EPS)).copy()
    used = {t: [] for t in TARGETS}
    for ti, t in enumerate(TARGETS):
        pool = pool_for(t, ti)
        if not pool:
            continue
        w = total_w / len(pool)
        for b in pool:
            ze[:, ti] += w * b["resid"][:, ti]
            used[t].append(b["name"])
    return sig(ze), used


STYLES = {"conservative": 0.10, "balanced": 0.25, "aggressive": 0.50}
overshoot_dir = anchor_dir  # Q2/Q3 overshoot reference direction


def diagnose(name, ens, used):
    print(f"\n================ {name} ensemble (total_w={STYLES[name]}) ================")
    sc = K.score_perrow(m, ens)
    dz = logit(np.clip(ens, EPS, 1 - EPS)) - logit(np.clip(anchor, EPS, 1 - EPS))
    print(f"  {'tgt':3s} | meanΔlogit vs anchor | CV anchor->ens (Δ) | failedDir corr")
    changed = []
    for ti, t in enumerate(TARGETS):
        md = float(dz[:, ti].mean()); mad = float(np.abs(dz[:, ti]).mean())
        cvd = sc[t] - sc_anch[t]
        # failed-direction overlap: corr of the ens-vs-anchor move with the overshoot direction (Q-overshoot family)
        ov = float(np.corrcoef(dz[:, ti], overshoot_dir[:, ti])[0, 1]) if dz[:, ti].std() > 1e-9 and overshoot_dir[:, ti].std() > 1e-9 else 0.0
        if mad > 1e-6:
            changed.append(t)
        print(f"  {t:3s} | {md:+.4f} (|{mad:.4f}|)      | {sc_anch[t]:.4f}->{sc[t]:.4f} ({cvd:+.5f}) | {ov:+.2f}"
              f"   used={len(used[t])}")
    dist = float(np.sqrt(((ens - anchor) ** 2).sum()))
    cv_mean_delta = K.meanll(sc) - K.meanll(sc_anch)
    print(f"  changed targets: {changed}")
    print(f"  prediction distance vs anchor (L2): {dist:.3f} | CV mean Δ vs anchor: {cv_mean_delta:+.5f}")
    # authoritative transfer gate
    cand = m[H.KEYS].copy()
    for ti, t in enumerate(TARGETS):
        cand[t] = ens[:, ti]
    r = vet_candidate(cand, label=f"G59_{name}", verbose=False)
    verds = {t: r["targets"][t]["verdict"] for t in TARGETS}
    pass_level = [t for t in TARGETS if verds[t] == "PASS_LEVEL"]
    print(f"  transfer gate verdicts: {verds}")
    print(f"  -> PASS_LEVEL (transferable) targets: {pass_level if pass_level else 'NONE'}")
    # CSV gate
    distinct = dist > 0.5
    cv_ok = cv_mean_delta < 0.003
    failed_overlap = any(abs(np.corrcoef(dz[:, ti], overshoot_dir[:, ti])[0, 1]) > 0.7
                         for ti in range(7) if dz[:, ti].std() > 1e-9 and overshoot_dir[:, ti].std() > 1e-9)
    transferable = len(pass_level) > 0
    gate = dict(distinct=distinct, cv_ok=cv_ok, non_failed_overlap=not failed_overlap, transferable=transferable)
    write = all(gate.values())
    print(f"  CSV GATE: {gate} -> WRITE={write}")
    return write, ens, changed


print("\n" + "#" * 70)
results = {}
for name in STYLES:
    ens, used = build_ens(STYLES[name])
    results[name] = diagnose(name, ens, used)

print("\n" + "#" * 70)
writers = [n for n in STYLES if results[n][0]]
if not writers:
    print("GOAL_059_RESULT = FAIL — no ensemble passes the CSV gate.")
    print("  (every diverse blend either is too close to anchor, or improves OOF CV but the transfer")
    print("   gate calls it LOTTERY/NO_GAIN = the documented CV-LB gap; none adds a PASS_LEVEL lever.)")
    print("  anchor 0.5615333471 유지. No CSV written.")
else:
    print(f"candidates passing gate: {writers} (would realize on test bank).")
print("\nDONE")
