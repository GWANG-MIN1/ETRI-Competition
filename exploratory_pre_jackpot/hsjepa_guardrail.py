"""Transfer-safe improvement on the 0.5677 anchor (h057 = kbs56774.csv): EXTREME-CELL GUARDRAIL.

Rationale (proper scoring rule / Regret-Allocator made real): the anchor was fit toward public
loss and contains recklessly-confident cells (literal 0/1, and 10-24% of S-cells >0.95). Under
logloss a single WRONG extreme cell is catastrophic (a 1e-6 miss costs ln(1e6)=13.8 -> +0.0079 to
mean logloss over 1750 cells). Clipping ONLY the extreme tail to a still-confident bound has:
  - tiny cost if the model is right there (e.g. 0.999->0.98 when label=1 costs ln(1.019)=0.019/cell)
  - large save if it is wrong (0.999->0.98 when label=0 saves ln(1000)-ln(50)=3.0/cell)
This is the safest possible edit on a hard-won anchor: it touches the catastrophic-exposure cells
only, leaves the discrimination intact, and bounds private-split downside.
"""
import numpy as np, pandas as pd
from pathlib import Path

ANCHOR = Path(r"C:\Users\박광민\Downloads\kbs56774 (2).csv")
RAW = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\outputs\raw")
OUT = Path(r"C:\Users\박광민\Documents\Codex\2026-05-26\https-dacon-io-competitions-official-236690\outputs\submissions")
TARGETS = ["Q1","Q2","Q3","S1","S2","S3","S4"]
KEYS = ["subject_id","sleep_date","lifelog_date"]

sample = pd.read_csv(RAW/"ch2026_submission_sample.csv")
anchor = pd.read_csv(ANCHOR)
anchor = sample[KEYS].merge(anchor, on=KEYS, how="left", validate="one_to_one")  # enforce sample order

P = anchor[TARGETS].to_numpy(float)
print("=== anchor extreme-cell exposure (per target, count of cells) ===")
for j,t in enumerate(TARGETS):
    p = P[:,j]
    print(f"  {t}: <0.02:{(p<0.02).sum():2d} <0.03:{(p<0.03).sum():2d} | >0.97:{(p>0.97).sum():2d} "
          f">0.98:{(p>0.98).sum():2d} >0.99:{(p>0.99).sum():2d} | ==0/1 exact:{((p<=1e-6)|(p>=1-1e-6)).sum()}")
tot98 = ((P<0.02)|(P>0.98)).sum(); tot97=((P<0.03)|(P>0.97)).sum()
print(f"  TOTAL cells outside [0.02,0.98]: {tot98} / 1750  | outside [0.03,0.97]: {tot97} / 1750")
# worst-case exposure: if every extreme cell were WRONG, logloss cost at 1e-6 clip
exact = ((P<=1e-6)|(P>=1-1e-6)).sum()
print(f"  exact 0/1 cells: {exact}  (each, if wrong, costs ln(1e6)/1750 = {np.log(1e6)/1750:.4f} to mean logloss)")

def validate(df):
    prob = df[TARGETS].to_numpy(float)
    return dict(rows=len(df), keys_match=bool(df[KEYS].equals(sample[KEYS])),
                dup=int(df[KEYS].duplicated().sum()), nan=int(np.isnan(prob).sum()),
                min=float(prob.min()), max=float(prob.max()),
                upload_safe=bool(len(df)==len(sample) and df[KEYS].equals(sample[KEYS])
                                 and not df[KEYS].duplicated().any() and np.isfinite(prob).all()
                                 and prob.min()>0.0 and prob.max()<1.0))

variants = {"guardrail_p985": (0.015, 0.985), "guardrail_p98": (0.02, 0.98), "guardrail_p97": (0.03, 0.97)}
for name,(lo,hi) in variants.items():
    out = anchor[KEYS].copy()
    Pc = np.clip(P, lo, hi)
    for j,t in enumerate(TARGETS):
        out[t] = Pc[:,j]
    changed = (np.abs(Pc-P)>1e-12).sum()
    d = np.abs(Pc-P)
    v = validate(out[KEYS+TARGETS] if False else out)
    print(f"\n--- {name}  clip[{lo},{hi}] ---")
    print(f"  changed cells: {changed}/1750 | mean|Δ|={d.mean():.5f} max|Δ|={d.max():.4f}")
    print(f"  per-target mean shift: " + " ".join(f"{t}={Pc[:,j].mean()-P[:,j].mean():+.4f}" for j,t in enumerate(TARGETS)))
    print(f"  upload-safe: {v['upload_safe']} (rows={v['rows']} keys={v['keys_match']} dup={v['dup']} "
          f"range[{v['min']:.4f},{v['max']:.4f}])")
    outp = OUT/f"submission_HSJEPA057_{name}.csv"
    if v["upload_safe"]:
        out[KEYS+TARGETS].to_csv(outp, index=False)
        print(f"  SAVED -> {outp.name}")
    else:
        print("  NOT SAVED")
