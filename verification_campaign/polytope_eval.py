#!/usr/bin/env python3
"""Reusable LB-relevant arbiter: evaluate a per-subject level move on FS against the
jackpot-aware label polytope (incl. the 0.5619 measurement).

Every base-model candidate must reduce to a per-(subject,target) logit shift on the FS
base; this module scores it: post-mean / q05 / q95 gain over the central posterior,
certified worst-case over the loose polytope, and 18-setting favorability.

Usage:
    import polytope_eval as PE
    PE.init()                                   # one-time setup (caches posterior)
    PE.eval_delta({('id05','Q2'): -0.3}, 'id05 down', settings18=True)
    PE.subjects, PE.TARGETS                      # available after init
    PE.overshoot_delta()                         # the 0.5619 anchor delta dict
Convention: gain < 0 = improvement (lower logloss). reference anchor (overshoot) ~ -0.0053.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ETRI = Path(r"C:\Users\박광민\Documents\Codex\etri_team")  # absolute (repo is portfolio mirror; data lives in the team workspace)
OLD = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\outputs\submissions")
RAW = ETRI / "data"
KEYS = ["subject_id", "sleep_date", "lifelog_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6; N_CELLS = 1750.0; H057_LB = 0.5677475939
FS_FILE = "submission_hsjepa_frontier_silence_positive_path_overshoot_sensor_1e013277_uploadsafe.csv"
ANCHOR_LB = 0.5619100863
_CACHE = HERE / "outputs" / "polytope_eval_cache.npz"

_S = None  # state dict


def _imp(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m


def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _lin(pf, pt):
    pf = np.clip(np.asarray(pf, float), EPS, 1 - EPS); pt = np.clip(np.asarray(pt, float), EPS, 1 - EPS)
    return (np.log((1 - pt) / (1 - pf)) - np.log(pt / pf), -np.log((1 - pt) / (1 - pf)),
            np.log(pt / (1 - pt)) - np.log(pf / (1 - pf)))


def init(force=False):
    global _S, subjects, TARGETS, FS, vidx
    if _S is not None and not force:
        return
    c1 = _imp(HERE / "candidate_1_public_loss_sparse_tomography.py", "c1_pe")
    CERT = _imp(Path(r"C:\Users\박광민\Downloads\human_state_drift_consistency_certifier.py"), "hc_pe")
    CERT.TRAIN_PATH = RAW / "ch2026_metrics_train.csv"; CERT.BASE_PATH = ETRI / FS_FILE

    def load_abs(p):
        return pd.read_csv(p).sort_values(KEYS).reset_index(drop=True)

    train, base = CERT.load_frames()
    drift = CERT.subject_half_drift(train)
    over_cfg = [c for c in CERT.CONFIGS if c.name == "drift_consistency_overshoot"][0]
    osteps = CERT.subject_steps(over_cfg, drift).set_index("subject_id")
    H057 = load_abs(c1.locate("submission_h057_q2row_fullvector_state_7cde1a77_uploadsafe.csv"))
    FSf = base.sort_values(KEYS).reset_index(drop=True)
    sub = H057["subject_id"].values; subs = sorted(set(sub))
    vix = {}; k = 0
    for t in TARGETS:
        for s in subs:
            vix[(s, t)] = k; k += 1
    n_var = k; masks = {s: sub == s for s in subs}; n_rows = {s: int(masks[s].sum()) for s in subs}
    fsub = FSf["subject_id"].values

    def pair_row(a, b):
        v = np.zeros(n_var); c0 = 0.0; sq = 0.0
        for t in TARGETS:
            coef, const, dlg = _lin(a[t].values, b[t].values); c0 += const.sum() / N_CELLS
            for s in subs:
                m = masks[s]; v[vix[(s, t)]] += coef[m].sum() / N_CELLS; sq += float(((dlg[m] - dlg[m].mean()) ** 2).sum())
        return v, c0, 0.5 * np.sqrt(sq) / N_CELLS

    # overshoot anchor delta + candidate frame
    d_over = np.zeros(n_var)
    for s in subs:
        d_over[vix[(s, "Q2")]] = float(osteps.loc[s, "Q2_logit_step"]); d_over[vix[(s, "Q3")]] = float(osteps.loc[s, "Q3_logit_step"])

    def cand(delta):
        df = FSf[KEYS].copy()
        for t in TARGETS:
            z = logit(FSf[t].values).copy()
            for s in subs:
                z[fsub == s] += delta[vix[(s, t)]]
            df[t] = sigmoid(z)
        return df

    named = []
    ledger = pd.read_csv(ETRI / "data_analytics" / "hsjepa_public_score_ledger.csv")
    for rec in ledger.to_dict("records"):
        p = c1.locate(str(rec["file"]))
        if p is None:
            continue
        v, c0, sg = pair_row(H057, load_abs(p)); named.append((v, c0, float(rec["public_lb"]) - H057_LB, sg))
    olds = {kk: load_abs(OLD / nm) for kk, nm in {
        "M": "submission_lb60247_to_pertarget_best_microblend.csv",
        "A": "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv",
        "B": "submission_Q1up100_plus_Q2recency_tau10_lam075.csv",
        "C": "submission_FINAL_Q2recency_plus_Q3recency.csv"}.items()}
    for f1, f2, d in [("A", "B", 0.5949167449 - 0.6001), ("B", "C", 0.593188787 - 0.5949167449), ("M", "A", 0.6001 - 0.60247)]:
        v, c0, sg = pair_row(olds[f1], olds[f2]); named.append((v, c0, d, sg))
    v_a, c0_a, sg_a = pair_row(H057, cand(d_over)); named.append((v_a, c0_a, ANCHOR_LB - H057_LB, sg_a))

    rate = {t: train.groupby("subject_id")[t].mean().to_dict() for t in TARGETS}
    Zfs = {t: {s: logit(FSf[t].values)[masks[s]] for s in subs} for t in TARGETS}

    def polytope(comp, boxw, Zz):
        bnd = [(max(0.02, rate[t][s] - boxw), min(0.98, rate[t][s] + boxw)) for t in TARGETS for s in subs]
        AA, bb = [], []
        for v, c0, d, sg in named:
            tol = 1e-5 + comp * abs(d) + Zz * sg
            AA.append(v); bb.append(d + tol - c0); AA.append(-v); bb.append(-(d - tol - c0))
        return np.array(AA), np.array(bb), bnd

    A_c, b_c, bnd_c = polytope(0.10, 0.35, 2.0)
    # posterior (cache)
    if _CACHE.exists() and not force:
        Ssamp = np.load(_CACHE)["S"]
    else:
        lob = np.array([b[0] for b in bnd_c]); hib = np.array([b[1] for b in bnd_c])
        Aall = np.vstack([A_c, np.eye(n_var), -np.eye(n_var)]); ball = np.concatenate([b_c, hib, -lob])
        nm = np.linalg.norm(Aall, axis=1, keepdims=True)
        che = linprog(np.r_[np.zeros(n_var), -1.0], A_ub=np.hstack([Aall, nm]), b_ub=ball,
                      bounds=[(None, None)] * n_var + [(0, None)], method="highs")
        x = che.x[:n_var]; rng = np.random.default_rng(2024); samp = []
        for step in range(40000):
            u = rng.standard_normal(n_var); u /= np.linalg.norm(u); t1, t0 = np.inf, -np.inf
            pos = u > 1e-12; neg = u < -1e-12
            if pos.any():
                t1 = min(t1, ((hib - x)[pos] / u[pos]).min()); t0 = max(t0, ((lob - x)[pos] / u[pos]).max())
            if neg.any():
                t1 = min(t1, ((lob - x)[neg] / u[neg]).min()); t0 = max(t0, ((hib - x)[neg] / u[neg]).max())
            au = A_c @ u; ax = A_c @ x; p2 = au > 1e-14; n2 = au < -1e-14
            if p2.any():
                t1 = min(t1, ((b_c - ax)[p2] / au[p2]).min())
            if n2.any():
                t0 = max(t0, ((b_c - ax)[n2] / au[n2]).max())
            if t1 <= t0:
                continue
            x = x + (t0 + (t1 - t0) * rng.random()) * u
            if step % 20 == 0 and step > 4000:
                samp.append(x.copy())
        Ssamp = np.array(samp); np.savez(_CACHE, S=Ssamp)

    _S = dict(subs=subs, vix=vix, n_var=n_var, n_rows=n_rows, Zfs=Zfs, named=named,
              rate=rate, polytope=polytope, S=Ssamp, d_over=d_over, osteps=osteps)
    globals()["subjects"] = subs
    return _S


def _gain_terms(delta):
    Zfs = _S["Zfs"]; n_rows = _S["n_rows"]; vix = _S["vix"]; subs = _S["subs"]
    A = 0.0; coefs = np.zeros(_S["n_var"])
    for t in TARGETS:
        for s in subs:
            d = delta[vix[(s, t)]]
            if d == 0:
                continue
            z = Zfs[t][s]
            A += float(-np.log((1 - sigmoid(z + d)) / (1 - sigmoid(z))).sum()) / N_CELLS
            coefs[vix[(s, t)]] = -n_rows[s] * d / N_CELLS
    return A, coefs


def _delta_from_dict(dd):
    d = np.zeros(_S["n_var"])
    for (s, t), val in dd.items():
        d[_S["vix"][(s, t)]] = float(val)
    return d


def overshoot_delta():
    init()
    return _S["d_over"].copy()


def eval_delta(delta, label="", over_base=False, settings18=False, verbose=True):
    """delta: dict {(subject,target):logit_shift} OR np.array(n_var). over_base=True adds
    the overshoot anchor first (evaluate move-on-top-of-anchor)."""
    init()
    d = _delta_from_dict(delta) if isinstance(delta, dict) else np.asarray(delta, float).copy()
    if over_base:
        d = d + _S["d_over"]
    A, coefs = _gain_terms(d); S = _S["S"]; g = S @ coefs + A
    A_l, b_l, bnd_l = _S["polytope"](0.30, 0.45, 3.0)
    r = linprog(-coefs, A_ub=A_l, b_ub=b_l, bounds=bnd_l, method="highs")
    worst = (A + (-r.fun)) if r.success else None
    out = dict(label=label, post_mean=float(g.mean()), q05=float(np.percentile(g, 5)),
               q95=float(np.percentile(g, 95)), worst_loose=worst, p_improve=float((g < 0).mean()))
    if settings18:
        cg = []
        for comp in (0.10, 0.20, 0.30):
            for boxw in (0.25, 0.35, 0.45):
                for Zz in (2.0, 3.0):
                    AA, bb, bnd = _S["polytope"](comp, boxw, Zz)
                    lob = np.array([b[0] for b in bnd]); hib = np.array([b[1] for b in bnd])
                    Aall = np.vstack([AA, np.eye(_S["n_var"]), -np.eye(_S["n_var"])]); ball = np.concatenate([bb, hib, -lob])
                    nmv = np.linalg.norm(Aall, axis=1, keepdims=True)
                    ch = linprog(np.r_[np.zeros(_S["n_var"]), -1.0], A_ub=np.hstack([Aall, nmv]), b_ub=ball,
                                 bounds=[(None, None)] * _S["n_var"] + [(0, None)], method="highs")
                    if ch.success:
                        cg.append(A + coefs @ ch.x[:_S["n_var"]])
        cg = np.array(cg); out["fav18"] = float(np.mean(cg < 0)); out["g18_range"] = (float(cg.min()), float(cg.max()))
    if verbose:
        extra = f" fav18={out.get('fav18','-')}" if settings18 else ""
        print(f"  [{label}] post-mean {out['post_mean']:+.5f} q05 {out['q05']:+.5f} "
              f"worst {out['worst_loose']:+.5f} P(<0) {out['p_improve']:.2f}{extra}")
    return out


if __name__ == "__main__":
    init()
    print(f"init ok. subjects={subjects}  posterior n={_S['S'].shape[0]}")
    eval_delta({}, "no-op (FS, should be ~0)")
    eval_delta(overshoot_delta(), "overshoot anchor (expect ~-0.0053)")
    eval_delta({("id05", "Q2"): -0.3}, "id05.Q2-0.3 over anchor", over_base=True, settings18=True)
