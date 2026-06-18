# Strategy-doc execution & verification (2026-06-18)

Source: `HS-JEPA_DACON236690_performance_strategy (1).md` (user-provided). The doc proposed
pivoting from "bigger JEPA" to `JEPA repr → low-dim state/gate → targetwise conservative
correction`, with 3 submissions: A=Q2/Q3 JEPA gate, B=S2/S4 sleep-proxy specialist (its #1
priority), C=pure calibration/shrinkage.

I executed every step on the project's honest harness (test_faithful + interleaved CV +
placebo + anchor comparison). Tools: `sleep_proxy/sp_00..sp_04`. Baseline reproduced:
unified OOF test_faithful mean **0.5986**; best public LB **0.5615** = FS + 0.8x Q2/Q3 recency.

## Decisive diagnostic (sp_00)
- **Night/sleep features are already subsumed by base.** The 5242-feature store has explicit
  night windows (`pre_sleep`, `sleep_00_03`, `sleep_03_06`) — 3337 sleep-named columns.
- **Where transferable room exists** (base test_faithful vs oracle per-subject constant):
  Q2 +0.0185, Q3 +0.0213, S2 +0.0187, S3 +0.0537 → base is OVER-confident per-row (room via
  level/shrinkage). S1/S4/Q1: base already beats subject-constant → **S4 has NO level room**
  (contradicts the doc's S4 priority).

## Step 1 / Submission B — S2/S4 night-proxy specialist (sp_02): **DEAD**
Built 17 night sleep-architecture proxies from the canvas (onset/wake/duration, WASO motion
bursts, phone-inactivity, HR slope/min, novel RMSSD), logit-blended a robust specialist onto base.
- S2, S3, S4: **strictly worse** than base at every blend weight, and worse than placebo. Subsumed.
- Only Q1 showed a tiny per-row gain (−0.0009, beats placebo) — per-row, won't transfer; not a doc target.

## Step 2 / Submission A — JEPA scalar gates (sp_03): **nothing new**
z_drift / z_anomaly / z_state as conservative logit gates on Q2/Q3/S2/S4.
- Only **z_drift on Q2** fires (−0.002…−0.005, sign 1.00, beats placebo) — but it overlaps the
  Q2 level already in best AND already failed LB transfer as a per-row blend (the 0.5677 submission).
- z_anomaly / z_state: indistinguishable from placebo. S2/S4: harmful.

## Step 3-4 / Submission C — calibration / shrinkage (sp_01, sp_04): **re-discovers best; rest dead**
- **subject-shrink → train mean**: robust gain on **Q2** (both proxies, beats placebo strongly),
  partial on Q3. But this is the Q2/Q3 level the **best already captures**.
- **S1/S3 shrink**: helps interleaved CV, **reverses on the future block** = date-bound mirage (won't transfer).
- **S2/S4 shrink, temperature scaling, probability clipping**: all dead (preds already in [0.12,0.88]).

## The CV-LB gap, proven (sp_04 anchor comparison)
The best (FS overshoot×0.8) scores **WORSE than raw base on honest CV** (Q3 +0.0086 test_faithful;
Q2/Q3 +0.021/+0.030 interleaved). Its LB win is a **global Q-up prior probe** that train labels
don't contain, so **no CV-measurable lever can beat it**. The only LB-mover is global prior
shifting, whose robust optimum is already at overshoot×0.8 (= 0.5615, memory `experiment_log_0614c`).

## Verdict (doc A/B/C)
All three doc directions, rigorously vetted, converge on the project's prior conclusion: the
transfer-safe lever is Q2/Q3 level (already in best); S/per-row/JEPA-gate moves are subsumed,
date-bound, or below the transfer floor. **No A/B/C submission beats 0.5615.** The doc's
reframing (gate + conservative blend) is methodologically sound but runs into the same statistical
wall (n=10, Q per-row non-transfer, base subsumes level/structure).

## ADDENDUM 2026-06-18b — the ONE live lever: two-bucket Q2/Q3 overshoot (sp_05, sp_06)
`sp_05_twobucket.py` (not covered above) tests a NEW degree of freedom the deployed best never
used. The best applies a UNIFORM sigma=0.8 to all 250 test rows. But the test = 62.4% interleaved
(dated between train days -> low drift -> wants small sigma) + 37.6% future (after all train days
-> max drift -> wants large sigma). The bucket label (row date > subject's max train date) is
**OBSERVED (zero estimation noise) => transferable LEVEL class**, Q2/Q3 only. So the best
over-shoots the interleaved majority.

`sp_06_twobucket_build.py` builds two-bucket TEST submissions and scores each on two arbiters:
- **Reconstruction exact**: build(0.8,0.8) reproduces the 0.5615 file to 1.1e-16; bucket split =
  94 future / 156 interleaved = 0.376/0.624 (matches the documented structure).
- **PE (polytope_eval) = LB-CALIBRATED, and OOS-validated**: it scores x0.8 better than x1.0 by
  −0.00037; the *real* LB difference (0.5615333 − 0.5619101) = −0.00038. x0.8 is NOT in the
  ledger, so this is genuine out-of-sample calibration (matches to 1e-5).
- **Two-bucket beats deployed-0.8 on BOTH arbiters** (forward-CV + PE) for many configs while
  several keep worst-loose negative (certified safe). First lever in 30+ approaches to pass
  transferable-class + forward-CV + LB-calibrated-arbiter simultaneously.

**Built candidates (NOT submitted):**
- **C1 (recommended, private-safe) = (int 0.6, fut 1.0)**: PE −0.00570 (−0.00048 vs deployed),
  worst-loose −0.00048, forward-CV 7t −0.0016 (seed signs 1.00). Predicted LB ≈ 0.5610.
  `submission_twobucket_int0p6_fut1p0_317adcf1_uploadsafe.csv`.
- **C2 (aggressive) = (int 0.4, fut 1.2)**: PE −0.00604 (−0.00082 vs deployed), worst-loose
  +0.00089 (small worst-case risk). Predicted LB ≈ 0.5607.
  `submission_twobucket_int0p4_fut1p2_50deaad1_uploadsafe.csv`.

**Honest bound:** gain is ~−0.0005 on LB. This is NOT a path to 0.54 (information floor ≈ 0.56
stands). It is a small, defensible, private-safe step below the current best — to be resolved only
by a real submission (offline is exhausted). Anchor best preserved = 0.5615.
