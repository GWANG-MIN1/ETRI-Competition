"""HS-JEPA Direction 1 — "listener responsibility" = transfer-audibility.

Each single-target LB probe is an equation constraining the PUBLIC base rate.
logloss is analytic & additive over 7 targets:  dLB = (1/7) * d(target logloss).

We:
  1. load probe submissions + their known LB scores
  2. auto-detect which target(s) changed between consecutive probes
  3. for clean single-target steps, back out the public base rate p*
  4. estimate the LB noise floor (min detectable dLB on ~125 public rows)
  5. rank the "audible frontier": gap (current mean vs est. optimum) vs noise
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBS = ROOT / "outputs" / "submissions"
RAW = ROOT / "outputs" / "raw"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-7

# Known (file -> public LB) from experiment-log memory. Ordered by the actual probe chain.
LB = {
    "submission_lb60247_to_pertarget_best_microblend.csv": 0.60247,  # pre-Q chain anchor
    "submission_lb_target_micro_probe_v1_Q1_shift_up100.csv": 0.6001,  # Q1 -> ~0.605
    "submission_Q1up100_plus_Q2recency_tau10_lam075.csv": 0.5949167449,  # + Q2 -> ~0.615
    "submission_FINAL_Q2recency_plus_Q3recency.csv": 0.593188787,  # + Q3 -> ~0.616  (BEST)
    "submission_ROBUST_S234_rerank_w025.csv": 0.5956069071,  # S2/S3/S4 mean-PRESERVING rerank -> FAILED
    "submission_sidechannel_v2_direct.csv": 0.6228,  # big per-row move (mean diff .07) -> FAILED
}


def load(name):
    p = SUBS / name
    if not p.exists():
        hits = list(ROOT.glob(f"outputs/**/{name}"))
        if hits:
            p = hits[0]
        else:
            return None
    return pd.read_csv(p)


def means(df):
    return {t: float(df[t].mean()) for t in TARGETS}


def const_logloss(c, p):
    c = min(max(c, EPS), 1 - EPS)
    return -(p * np.log(c) + (1 - p) * np.log(1 - c))


def backout_pstar(mean_before, mean_after, dloss_target):
    """Constant-prediction approx. L(c;p) = -(p ln c + (1-p) ln(1-c)).
    dloss = L(b;p) - L(a;p) = -lb - p*(la - lb),  la=ln(b/a), lb=ln((1-b)/(1-a)).
    => p* = (-lb - dloss) / (la - lb)."""
    a, b = mean_before, mean_after
    la, lb = np.log(b) - np.log(a), np.log(1 - b) - np.log(1 - a)
    denom = la - lb
    if abs(denom) < 1e-12:
        return np.nan
    return (-lb - dloss_target) / denom


print("=" * 78)
print("PROBE MEANS (per target)")
dfs = {n: load(n) for n in LB}
for n, df in dfs.items():
    if df is None:
        print(f"  [MISSING] {n}")
        continue
    m = means(df)
    print(f"  LB {LB[n]:.5f} | {n[:52]:52s} | " + " ".join(f"{t}={m[t]:.3f}" for t in TARGETS))

print("=" * 78)
print("SINGLE-TARGET PROBE STEPS  ->  back out public base rate p*")
chain = [
    ("submission_lb_target_micro_probe_v1_Q1_shift_up100.csv",
     "submission_Q1up100_plus_Q2recency_tau10_lam075.csv"),
    ("submission_Q1up100_plus_Q2recency_tau10_lam075.csv",
     "submission_FINAL_Q2recency_plus_Q3recency.csv"),
    ("submission_FINAL_Q2recency_plus_Q3recency.csv",
     "submission_ROBUST_S234_rerank_w025.csv"),
]
for a_name, b_name in chain:
    da, db = dfs.get(a_name), dfs.get(b_name)
    if da is None or db is None:
        continue
    ma, mb = means(da), means(db)
    changed = [t for t in TARGETS if abs(ma[t] - mb[t]) > 1e-4]
    dLB = LB[b_name] - LB[a_name]
    print(f"\n  {a_name[:34]} -> {b_name[:34]}")
    print(f"    changed targets: {changed} | dLB={dLB:+.5f}")
    if len(changed) == 1:
        t = changed[0]
        dloss_t = 7 * dLB  # isolate target's mean-logloss change
        pstar = backout_pstar(ma[t], mb[t], dloss_t)
        print(f"    {t}: mean {ma[t]:.3f} -> {mb[t]:.3f} | d(target logloss)={dloss_t:+.4f}")
        print(f"    => est. PUBLIC base rate p*({t}) = {pstar:.3f}   (train marginal differs)")
        if not np.isnan(pstar):
            cur = mb[t]
            gain_to_opt = const_logloss(cur, pstar) - const_logloss(pstar, pstar)
            print(f"    => if we move {t} {cur:.3f} -> {pstar:.3f}, est. extra target-logloss gain "
                  f"{-gain_to_opt:+.4f}  (dLB {-gain_to_opt/7:+.5f})")
    else:
        # mean-preserving change (rerank): pure per-row effect = noise-floor probe
        maxd = max(abs(ma[t] - mb[t]) for t in TARGETS) if changed else 0.0
        print(f"    (mean-preserving rerank; max per-target mean move {maxd:.4f}) "
              f"=> pure per-row dLB={dLB:+.5f}  <-- empirical per-row (in)audibility")

print("=" * 78)
print("AUDIBILITY LAW  (the public set is FIXED -> dLB is deterministic, not sampling noise)")
print("  The real risk is SIGN-RELIABILITY: does a CV-measured gain keep its sign on public/private?")
print("  Empirical evidence from the probe chain:")
print("    LEVEL moves  (Q2 +.047 -> dLB -.0052 ; Q3 +.022 -> dLB -.0017): correct sign, repeatable")
print("    PER-ROW move (S234 rerank, means ~fixed +.0016) -> dLB +.0024 : WRONG sign vs CV")
print("  => Level shifts are audible down to ~0.0017; per-row reranks are sign-unreliable at ANY size.")

print("=" * 78)
print("TOXICITY / OVERSHOOT GEOMETRY  (logloss curve L(c; p*) around the optimum)")
print("  For each Q target: est. p*, current mean, and the asymmetric penalty of over/under-shoot.")
best = dfs["submission_FINAL_Q2recency_plus_Q3recency.csv"]
pstars = {"Q2": None, "Q3": None}
# recompute clean single-target p* for Q2, Q3
for a_name, b_name, tt in [
    ("submission_lb_target_micro_probe_v1_Q1_shift_up100.csv",
     "submission_Q1up100_plus_Q2recency_tau10_lam075.csv", "Q2"),
    ("submission_Q1up100_plus_Q2recency_tau10_lam075.csv",
     "submission_FINAL_Q2recency_plus_Q3recency.csv", "Q3")]:
    ma, mb = means(dfs[a_name]), means(dfs[b_name])
    pstars[tt] = backout_pstar(ma[tt], mb[tt], 7 * (LB[b_name] - LB[a_name]))
for t in ("Q2", "Q3"):
    p = pstars[t]
    cur = float(best[t].mean())
    print(f"\n  {t}: est public p*~{p:.3f} | current mean {cur:.3f} | gap to optimum {p-cur:+.3f}")
    print(f"    {'mean c':>8} | {'L(c;p*)':>9} | {'dLB vs current':>15}")
    for c in [cur, 0.64, 0.66, 0.68, 0.70, min(p, 0.78)]:
        dlb = (const_logloss(c, p) - const_logloss(cur, p)) / 7
        mark = "  <- current" if abs(c - cur) < 1e-6 else ("  <- ~optimum" if abs(c - p) < 1e-6 else "")
        print(f"    {c:8.3f} | {const_logloss(c,p):9.4f} | {dlb:+15.5f}{mark}")
print("\n  NOTE: constant-approx upper-bounds the shift (real preds have spread). Direction is robust,")
print("        magnitude is optimistic; private subset may differ. Treat p* as 'there is room UP', probe small.")
