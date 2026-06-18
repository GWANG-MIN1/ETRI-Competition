"""Strategy-doc Step 2 + Submission A: compress JEPA z[96] -> interpretable scalar gates,
test each for transferable gain on Q2/Q3/S2/S4 (kill if not beating placebo).

z scalars from cached circa-JEPA embedding (worldmodel_jepa/cache/wm_circajepa_z.npz):
  z_drift   = within-subject time-direction projection (reuse cached wm_drift_feature)
  z_anomaly = distance from subject's centroid (per-row novelty)
  z_state   = KMeans(z,K=2) cluster id (per-row regime)
Each is built per-row, recency/within-subject standardized, then blended conservatively
(logit, small alpha) onto base ONLY for Q2/Q3/S2/S4 and scored honestly vs base + placebo
(shuffled scalar). Also run the authoritative transfer gate on the best per-target alpha.
NO LB fitting.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

KIT_DIR = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
WM = Path(__file__).resolve().parents[1] / "worldmodel_jepa"
sys.path.insert(0, str(HSJEPA_SRC)); sys.path.insert(0, str(KIT_DIR)); sys.path.insert(0, str(WM))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa

EPS = 1e-6
TARGETS = list(H.TARGETS)
GATE_T = ["Q2", "Q3", "S2", "S4"]
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173]


def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))


def build_scalars(m):
    zd = np.load(WM / "cache" / "wm_circajepa_z.npz", allow_pickle=True)
    z = zd["z"]  # (700,96)
    keys = zd["keys"] if "keys" in zd.files else None
    # canvas keys give alignment
    cv = np.load(WM / "cache" / "wm_canvas.npz", allow_pickle=True)
    ck = cv["keys"]  # (700,2)
    kmap = {(s, pd.to_datetime(d).normalize()): i for i, (s, d) in enumerate(ck)}
    ridx = np.array([kmap[(s, pd.to_datetime(d).normalize())] for s, d in zip(m.subject_id, m.lifelog_date)])
    zt = z[ridx]  # (450,96)
    subj = m["subject_id"].values
    # z_anomaly: distance from subject centroid
    anomaly = np.zeros(len(m))
    for s in np.unique(subj):
        mk = subj == s
        c = zt[mk].mean(0)
        anomaly[mk] = np.linalg.norm(zt[mk] - c, axis=1)
    # z_state: KMeans K=2 on full z
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(zt)
    state = km.labels_.astype(float)
    # z_drift: cached drift feature
    dd = np.load(WM / "cache" / "wm_drift_feature.npz", allow_pickle=True)
    fkeys = dd["keys"]
    fmap = {(s, pd.to_datetime(d).normalize()): dd["f"][i] for i, (s, d) in enumerate(fkeys)}
    drift = np.array([fmap.get((s, pd.to_datetime(d).normalize()), 0.0)
                      for s, d in zip(m.subject_id, m.lifelog_date)])
    return {"z_drift": drift, "z_anomaly": anomaly, "z_state": state}


def zscore_within_subject(m, v):
    out = v.astype(float).copy()
    subj = m["subject_id"].values
    for s in np.unique(subj):
        mk = subj == s
        mu = out[mk].mean(); sd = out[mk].std() + 1e-9
        out[mk] = (out[mk] - mu) / sd
    return out


def run(proxy="test_faithful"):
    m = K.base()
    sca = build_scalars(m)
    alphas = [0.03, 0.05, 0.1]
    print(f"\n================ JEPA scalar gates [{proxy}] ================")
    for name, raw in sca.items():
        v = zscore_within_subject(m, raw)
        res = {t: {a: [] for a in alphas} for t in GATE_T}
        for t in GATE_T:
            res[t]["placebo"] = []
        for sd in SEEDS:
            if proxy == "test_faithful":
                held = H.test_faithful_mask(m, sd); splits = [(np.flatnonzero(~held), np.flatnonzero(held))]
            else:
                splits = H.interleaved_folds(m, 5, sd)
            rng = np.random.default_rng(sd)
            vpl = v.copy()
            subj = m["subject_id"].values
            for s in np.unique(subj):
                mk = np.flatnonzero(subj == s); vpl[mk] = vpl[rng.permutation(mk)]
            for t in GATE_T:
                y = m[f"y_{t}"].values
                # fit sign/scale of gate on TRAIN fold via correlation of v with residual
                for tr, va in splits:
                    bb = m[f"p_{t}"].values
                    # direction: regress (y-p) on v over train -> coef sign
                    resid = y[tr] - bb[tr]
                    coef = np.polyfit(v[tr], resid, 1)[0] if v[tr].std() > 1e-9 else 0.0
                    base_ll = H.bll(y[va], bb[va])
                    for a in alphas:
                        z = logit(bb[va]) + a * np.sign(coef) * v[va]
                        res[t][a].append(H.bll(y[va], sig(z)) - base_ll)
                    zp = logit(bb[va]) + 0.05 * np.sign(coef) * vpl[va]
                    res[t]["placebo"].append(H.bll(y[va], sig(zp)) - base_ll)
        print(f"\n  -- gate={name} --")
        print("    tgt | " + " | ".join([f"a={a}" for a in alphas]) + " | placebo")
        for t in GATE_T:
            cells = [f"{np.mean(res[t][a]):+.5f}({np.mean(np.array(res[t][a])<0):.2f})" for a in alphas]
            pl = np.mean(res[t]["placebo"])
            print(f"    {t}  | " + " | ".join(cells) + f" | {pl:+.5f}")


if __name__ == "__main__":
    for proxy in ("test_faithful", "interleaved"):
        run(proxy)
    print("\nDONE")
