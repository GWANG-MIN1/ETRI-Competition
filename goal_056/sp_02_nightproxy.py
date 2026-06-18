"""Strategy-doc Step 1 + Submission B: S2/S4 sleep-proxy specialist.

Build night-window sleep-ARCHITECTURE proxies (derived quantities the fixed-window
feature store may not hold: estimated sleep onset/wake/duration, WASO motion bursts,
phone-inactivity, HR slope/min, novel RMSSD) from the cached circadian canvas, then ask
the decisive question: does a conservative specialist on these proxies add anything BEYOND
the 5242-feature base (which already has pre_sleep/sleep_00_03/sleep_03_06 aggregates)?

Honest test (per target, per seed, test_faithful + interleaved):
  specialist = LogisticRegression on standardized night proxies (robust, small data)
  evaluate: (i) logit-blend(base, specialist, w) held-out delta vs base, w in {.05,.1,.2,.3}
            (ii) PLACEBO specialist on column-shuffled proxies (kills row<->day pairing)
A real structural gain must beat base AND beat placebo on held-out. NO LB fitting.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

KIT_DIR = Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
HSJEPA_SRC = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
CANVAS = Path(__file__).resolve().parents[1] / "worldmodel_jepa" / "cache" / "wm_canvas.npz"
sys.path.insert(0, str(HSJEPA_SRC)); sys.path.insert(0, str(KIT_DIR))
import goal054_kit as K  # noqa
import hsjepa_core as H  # noqa

EPS = 1e-6
TARGETS = list(H.TARGETS)
SEEDS = [11, 23, 37, 51, 67, 83, 101, 131, 151, 173]


def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1.0 / (1.0 + np.exp(-z))


def hbin(h):
    hh = h if h >= 12 else h + 24
    return int((hh - 12) * 12)


def build_night_features(n_tr_only=True):
    d = np.load(CANVAS, allow_pickle=True)
    X = d["X"]; obs = d["obs"]; chans = list(d["chans"]); n_tr = int(d["n_tr"])
    keys = d["keys"]  # (N,2) str subject_id, lifelog_date
    N = X.shape[0]
    ci = {c: i for i, c in enumerate(chans)}
    nb = slice(hbin(22), hbin(33))  # 22:00 -> 09:00
    bins = np.arange(nb.start, nb.stop)
    feats = {}

    def covered_series(row, ch):
        v = X[row, ci[ch], nb]; o = obs[row, ci[ch], nb] > 0
        return v, o

    names = ["n_hr_mean", "n_hr_min", "n_hr_slope", "n_rmssd_mean", "n_rmssd_min",
             "n_active_mean", "n_motion_burst", "n_step_sum", "waso_proxy",
             "onset_bin", "wake_bin", "sleep_dur", "phone_inact_dur",
             "unlock_count", "n_light_mean", "charge_frac", "hr_cover_night"]
    F = np.full((N, len(names)), np.nan, dtype="float64")
    for r in range(N):
        # HR-based
        hr, hro = covered_series(r, "hr_mean")
        if hro.sum() >= 3:
            F[r, 0] = hr[hro].mean(); F[r, 1] = hr[hro].min()
            xx = bins[hro].astype(float); yy = hr[hro]
            F[r, 2] = np.polyfit(xx - xx.mean(), yy, 1)[0]
        rm, rmo = covered_series(r, "hr_rmssd")
        if rmo.sum() >= 3:
            F[r, 3] = rm[rmo].mean(); F[r, 4] = rm[rmo].min()
        F[r, 16] = (obs[r, ci["hr_mean"], nb] > 0).mean()
        # motion / activity (active_rate is z-scored; use threshold on z>0.5 as 'moving')
        act = X[r, ci["active_rate"], nb]
        acto = obs[r, ci["active_rate"], nb] > 0
        if acto.sum() >= 3:
            F[r, 5] = act[acto].mean()
            mv = (act > 0.5) & acto
            F[r, 6] = int(mv.sum())
            # WASO proxy: motion bursts strictly inside the inferred sleep block
        step = X[r, ci["step"], nb]; stepo = obs[r, ci["step"], nb] > 0
        if stepo.sum() >= 1:
            F[r, 7] = step[stepo & (step > 0)].sum()
        # screen-based sleep timing
        scr = X[r, ci["screen_on"], nb]; scro = obs[r, ci["screen_on"], nb] > 0
        on = (scr > 0.0) & scro  # any screen activity (z>0 means above-mean usage)
        quiet = scro & (~on)
        if scro.sum() >= 5:
            # onset = first index after which a long quiet run begins; wake = last quiet->on
            qi = quiet.astype(int)
            # longest quiet run = phone inactivity (sleep block proxy)
            best_len = best_s = best_e = 0; cur_s = None; cur = 0
            for k in range(len(qi)):
                if qi[k]:
                    if cur_s is None:
                        cur_s = k
                    cur += 1
                    if cur > best_len:
                        best_len = cur; best_s = cur_s; best_e = k
                else:
                    cur = 0; cur_s = None
            F[r, 9] = np.nan  # waso filled below
            F[r, 12] = best_len
            F[r, 13] = int(np.sum((scr[1:] > 0) & ~(scr[:-1] > 0) & scro[1:]))  # unlock edges
            if best_len >= 6:  # >=30 min quiet -> treat as sleep block
                F[r, 9] = best_s + nb.start; F[r, 10] = best_e + nb.start
                F[r, 11] = best_len
                # WASO proxy: motion bursts inside the sleep block window
                blk = np.arange(best_s, best_e + 1)
                if acto[blk].sum() > 0:
                    F[r, 8] = int(((act[blk] > 0.5) & acto[blk]).sum())
        lt = X[r, ci["light"], nb]; lto = obs[r, ci["light"], nb] > 0
        if lto.sum() >= 3:
            F[r, 14] = lt[lto].mean()
        ch = X[r, ci["charging"], nb]; cho = obs[r, ci["charging"], nb] > 0
        if cho.sum() >= 3:
            F[r, 15] = ch[cho].mean()
    df = pd.DataFrame(F, columns=names)
    df["subject_id"] = keys[:, 0]
    df["lifelog_date"] = pd.to_datetime(keys[:, 1])
    if n_tr_only:
        df = df.iloc[:n_tr].reset_index(drop=True)
    return df, names


def align_to_base(m, df, names):
    merged = m[["subject_id", "lifelog_date"]].merge(df, on=["subject_id", "lifelog_date"], how="left")
    Xf = merged[names].to_numpy("float64")
    return Xf


def impute_and_scale(Xtr, Xva):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xva = np.where(np.isfinite(Xva), Xva, med)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xva - mu) / sd


def run(proxy="test_faithful"):
    m = K.base()
    df, names = build_night_features(n_tr_only=True)
    Xf = align_to_base(m, df, names)
    cover = np.isfinite(Xf).mean(0)
    print(f"\n[{proxy}] night-proxy coverage per feature:")
    for nm, cv in zip(names, cover):
        print(f"    {nm:16s} {cv:.3f}")
    ws = [0.05, 0.1, 0.2, 0.3]
    # results: per target -> {w: [deltas]}, placebo
    res = {t: {("blend", w): [] for w in ws} for t in TARGETS}
    for t in TARGETS:
        res[t]["placebo"] = []
    for sd in SEEDS:
        if proxy == "test_faithful":
            held = H.test_faithful_mask(m, sd)
            splits = [(np.flatnonzero(~held), np.flatnonzero(held))]
        else:
            splits = H.interleaved_folds(m, 5, sd)
        rng = np.random.default_rng(sd)
        Xpl = Xf.copy()
        for j in range(Xpl.shape[1]):
            Xpl[:, j] = Xpl[rng.permutation(len(Xpl)), j]
        for t in TARGETS:
            y = m[f"y_{t}"].values
            for tr, va in splits:
                Xtr, Xva = impute_and_scale(Xf[tr], Xf[va])
                clf = LogisticRegression(C=0.3, max_iter=2000)
                clf.fit(Xtr, y[tr])
                sp = clf.predict_proba(Xva)[:, 1]
                bb = m[f"p_{t}"].values[va]
                base_ll = H.bll(y[va], bb)
                for w in ws:
                    pb = sig((1 - w) * logit(bb) + w * logit(sp))
                    res[t][("blend", w)].append(H.bll(y[va], pb) - base_ll)
                # placebo at w=0.2
                Xtrp, Xvap = impute_and_scale(Xpl[tr], Xpl[va])
                clp = LogisticRegression(C=0.3, max_iter=2000).fit(Xtrp, y[tr])
                spp = clp.predict_proba(Xvap)[:, 1]
                pbp = sig((1 - 0.2) * logit(bb) + 0.2 * logit(spp))
                res[t]["placebo"].append(H.bll(y[va], pbp) - base_ll)
    print(f"\n=== night-proxy specialist blend deltas [{proxy}] (delta<0 = better than base) ===")
    print("  tgt | " + " | ".join([f"w={w:>4}" for w in ws]) + " | placebo(w.2)")
    for t in TARGETS:
        cells = []
        for w in ws:
            d = np.array(res[t][("blend", w)]); cells.append(f"{d.mean():+.5f}({np.mean(d<0):.2f})")
        pl = np.array(res[t]["placebo"])
        flag = ""
        bw = min(ws, key=lambda w: np.mean(res[t][("blend", w)]))
        bd = np.mean(res[t][("blend", bw)])
        if bd < -0.0008 and np.mean(np.array(res[t][("blend", bw)]) < 0) >= 0.7 and bd < pl.mean():
            flag = f"  <<< (best w={bw})"
        print(f"  {t}  | " + " | ".join(cells) + f" | {pl.mean():+.5f}{flag}")


if __name__ == "__main__":
    for proxy in ("test_faithful", "interleaved"):
        run(proxy)
    print("\nDONE")
