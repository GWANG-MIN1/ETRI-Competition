import sys; from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
import lightgbm as lgb
sys.path.insert(0, r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs")
sys.path.insert(0, r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\src")
import goal054_kit as K, hsjepa_core as H
# rebuild features exactly as goal063
WMC=Path(r"C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\worldmodel_jepa\cache")
def hbin(h): hh=h if h>=12 else h+24; return int((hh-12)*12)
m=K.base(); subj=m["subject_id"].values
d=np.load(WMC/"wm_canvas.npz",allow_pickle=True); X=d["X"];obs=d["obs"];chans=list(d["chans"]);keys=d["keys"]
cmap={(s,pd.to_datetime(dt).normalize()):i for i,(s,dt) in enumerate(keys)}
ridx=np.array([cmap[(s,pd.to_datetime(dd).normalize())] for s,dd in zip(m.subject_id,m.lifelog_date)])
nb=slice(hbin(22),hbin(33)); ci={c:i for i,c in enumerate(chans)}; F={}
for c in chans: F[f"cov_{c}"]=np.array([(obs[r,ci[c],nb]>0).mean() for r in ridx])
for c in [c for c in ("hr_mean","active_rate","screen_on","light","step","hr_rmssd") if c in ci]:
    nm=np.array([X[r,ci[c],nb][obs[r,ci[c],nb]>0].mean() if (obs[r,ci[c],nb]>0).any() else np.nan for r in ridx]); F[f"nm_{c}"]=nm
    dev=np.full(len(m),np.nan)
    for s in np.unique(subj):
        mk=subj==s;v=nm[mk];med=np.nanmedian(v);iqr=(np.nanpercentile(v,75)-np.nanpercentile(v,25)) or 1.0;dev[mk]=(v-med)/iqr
    F[f"sdev_{c}"]=dev
R=np.column_stack(list(F.values())).astype(float); cm=np.nanmedian(R,0);cm=np.where(np.isfinite(cm),cm,0.); R=np.where(np.isfinite(R),R,cm)
z=np.load(WMC/"wm_circajepa_z.npz",allow_pickle=True)["z"][ridx]; Rz=np.column_stack([R,z])
LGB=dict(n_estimators=200,learning_rate=0.04,num_leaves=15,min_child_samples=25,reg_lambda=8.0,subsample=0.8,colsample_bytree=0.7,n_jobs=-1,verbose=-1)
SEEDS=[11,23,37,51,67,83,101,131,151,173]
def fut(seed,ff=0.15):
    rng=np.random.default_rng(seed);f=np.zeros(len(m),bool)
    for s,idx in m.groupby("subject_id").groups.items():
        idx=np.array(sorted(idx,key=lambda i:m.at[i,"sleep_date"]));f[idx[-max(1,int(round(len(idx)*ff))):]]=True
    return f
def perseed(Feat,t):
    r=(m[f"y_{t}"]-m[f"p_{t}"]).values; cs=[]
    for seed in SEEDS:
        held=H.test_faithful_mask(m,seed);tr=np.flatnonzero(~held);va=np.flatnonzero(held)
        pr=lgb.LGBMRegressor(**LGB,random_state=seed).fit(Feat[tr],r[tr]).predict(Feat[va])
        vf=np.intersect1d(va,np.flatnonzero(fut(seed)));pos=np.searchsorted(va,vf)
        if len(vf)>8 and np.std(pr[pos])>1e-9: cs.append(pearsonr(pr[pos],r[vf])[0])
    return np.array(cs)
def placebo_null(Feat,t,n=30):
    r=(m[f"y_{t}"]-m[f"p_{t}"]).values; out=[]
    for k in range(n):
        rng=np.random.default_rng(1000+k); rp=r[rng.permutation(len(r))]; cs=[]
        for seed in SEEDS[:5]:
            held=H.test_faithful_mask(m,seed);tr=np.flatnonzero(~held);va=np.flatnonzero(held)
            pr=lgb.LGBMRegressor(**LGB,random_state=seed).fit(Feat[tr],rp[tr]).predict(Feat[va])
            vf=np.intersect1d(va,np.flatnonzero(fut(seed)));pos=np.searchsorted(va,vf)
            if len(vf)>8 and np.std(pr[pos])>1e-9: cs.append(pearsonr(pr[pos],r[vf])[0])
        out.append(np.mean(cs))
    return np.array(out)
for tag,Feat,t in [("S3 risk-only",R,"S3"),("Q2 risk+z",Rz,"Q2"),("S3 risk+z",Rz,"S3"),("Q2 risk-only",R,"Q2")]:
    cs=perseed(Feat,t); null=placebo_null(Feat,t)
    print(f"{tag}: future corr mean {cs.mean():+.3f} | pos seeds {int((cs>0).sum())}/{len(cs)} | "
          f"placebo null mean {null.mean():+.3f} sd {null.std():.3f} 95pct {np.percentile(null,95):+.3f} | "
          f"real>95pct null: {cs.mean()>np.percentile(null,95)}")
