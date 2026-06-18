import sys; from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
sys.path.insert(0, r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
sys.path.insert(0, r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
import goal054_kit as K, hsjepa_core as H
from gate_transfer_vet import vet_candidate
WMC=Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\worldmodel_jepa\cache")
EPS=1e-6; TARGETS=list(K.TARGETS); SEEDS=[11,23,37,51,67,83,101,131,151,173]
def clip01(p): return np.clip(p,EPS,1-EPS)
def logit(p): return np.log(clip01(p)/(1-clip01(p)))
def sig(z): return 1/(1+np.exp(-z))
m=K.base(); subj=m["subject_id"].values
RES={t:(m[f"y_{t}"].values-m[f"p_{t}"].values) for t in TARGETS}
df=np.load(WMC/"wm_drift_feature.npz",allow_pickle=True); keys=df["keys"]; f700=df["f"]
cmap={(s,pd.to_datetime(dt).normalize()):i for i,(s,dt) in enumerate(keys)}
ridx=np.array([cmap[(s,pd.to_datetime(dd).normalize())] for s,dd in zip(m.subject_id,m.lifelog_date)])
drift=f700[ridx]; dz=(drift-drift.mean())/(drift.std()+1e-9)
def fut(seed,ff=0.15):
    rng=np.random.default_rng(seed);f=np.zeros(len(m),bool)
    for s,idx in m.groupby("subject_id").groups.items():
        idx=np.array(sorted(idx,key=lambda i:m.at[i,"sleep_date"]));f[idx[-max(1,int(round(len(idx)*ff))):]]=True
    return f

# --- strict placebo null for M2 drift (per target): shuffle drift 30x, future corr distribution ---
def strict_drift(t):
    r=RES[t]; real=[]; 
    for seed in SEEDS:
        fm=fut(seed)
        if np.std(dz[fm])>1e-9 and np.std(r[fm])>1e-9: real.append(pearsonr(dz[fm],r[fm])[0])
    real=np.mean(real); pos=np.mean([np.sign(x) for x in [real]])
    null=[]
    for k in range(30):
        rng=np.random.default_rng(700+k); dp=dz[rng.permutation(len(dz))]; cs=[]
        for seed in SEEDS[:5]:
            fm=fut(seed)
            if np.std(dp[fm])>1e-9 and np.std(r[fm])>1e-9: cs.append(pearsonr(dp[fm],r[fm])[0])
        null.append(np.mean(cs))
    null=np.array(null)
    return real, null.mean(), np.percentile(np.abs(null),95)

# --- gate test: build directional-correction candidate (only target t) and vet ---
def gate_drift(t):
    # slope = OLS resid~dz on ALL train (deployment); apply logit shift
    r=RES[t]; slope=np.polyfit(dz, r, 1)[0]
    cand=m[H.KEYS].copy()
    for tt in TARGETS: cand[tt]=m[f"p_{tt}"].values
    cand[t]=sig(logit(m[f"p_{t}"].values)+slope*dz)  # small linear directional nudge
    rr=vet_candidate(cand,label=f"M2_{t}",verbose=False)
    return rr["targets"][t]["verdict"]
def gate_subj(t):  # M1 per-subject relevel by train rbar (deployment=all train)
    r=RES[t]; rbar={s:float(r[subj==s].mean()) for s in np.unique(subj)}
    cand=m[H.KEYS].copy()
    for tt in TARGETS: cand[tt]=m[f"p_{tt}"].values
    cand[t]=clip01(m[f"p_{t}"].values+np.array([rbar[s] for s in subj]))
    rr=vet_candidate(cand,label=f"M1_{t}",verbose=False)
    return rr["targets"][t]["verdict"]

print("=== STRICT CONFIRM of red-team breaks (gate is the date-bound vs transferable arbiter) ===")
print("M2 drift candidates:")
for t in ("Q1","Q2","S2","S4"):
    real,nmean,n95=strict_drift(t); v=gate_drift(t)
    beats=abs(real)>n95
    print(f"  {t}: future corr {real:+.3f} | placebo|95%| {n95:.3f} | beats_placebo {beats} | GATE={v}")
print("M1 subject-relevel candidates:")
for t in ("S1",):
    v=gate_subj(t); print(f"  {t}: GATE={v}")

# --- M3 redo with FIXED placebo (finite-masked) ---
print("\n=== M3 (fixed placebo): signed sensor-regime g vs residual, FUTURE ===")
def hbin(h): hh=h if h>=12 else h+24; return int((hh-12)*12)
d=np.load(WMC/"wm_canvas.npz",allow_pickle=True); X=d["X"];obs=d["obs"];chans=list(d["chans"]);ci={c:i for i,c in enumerate(chans)};nb=slice(hbin(22),hbin(33))
def nm(ch): return np.array([X[r,ci[ch],nb][obs[r,ci[ch],nb]>0].mean() if (obs[r,ci[ch],nb]>0).any() else np.nan for r in ridx])
md=m.copy(); md["_d"]=pd.to_datetime(md["lifelog_date"])
def srd(ch):
    v=nm(ch); g=np.full(len(m),np.nan)
    for s in np.unique(subj):
        idx=np.flatnonzero(subj==s); order=idx[np.argsort(md["_d"].values[idx])]
        for k,i in enumerate(order):
            prev=order[max(0,k-7):k]
            if len(prev)>=2 and np.isfinite(v[i]): b=np.nanmedian(v[prev]); sd=np.nanstd(v[prev]) or 1.0; g[i]=(v[i]-b)/sd
    return g
G={c:srd(c) for c in ("hr_mean","active_rate","screen_on") if c in ci}
for t in TARGETS:
    r=RES[t]; best=None
    for gn,g in G.items():
        ok=np.isfinite(g); real=[]
        for seed in SEEDS:
            fm=fut(seed)&ok
            if fm.sum()>=12 and np.std(g[fm])>1e-9 and np.std(r[fm])>1e-9: real.append(pearsonr(g[fm],r[fm])[0])
        if not real: continue
        real=np.mean(real)
        null=[]
        for k in range(20):
            rng=np.random.default_rng(900+k); idxp=rng.permutation(np.flatnonzero(ok)); gp=g.copy(); gp[ok]=g[idxp]; cs=[]
            for seed in SEEDS[:5]:
                fm=fut(seed)&ok
                if fm.sum()>=12 and np.std(gp[fm])>1e-9 and np.std(r[fm])>1e-9: cs.append(pearsonr(gp[fm],r[fm])[0])
            if cs: null.append(np.mean(cs))
        n95=np.percentile(np.abs(null),95) if null else np.nan
        if best is None or abs(real)>abs(best[1]): best=(gn,real,n95)
    gn,real,n95=best
    print(f"  {t}: g={gn:11s} future corr {real:+.3f} | placebo|95%| {n95:.3f} | beats {abs(real)>n95}")
