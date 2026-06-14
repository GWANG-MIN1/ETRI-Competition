# Verification Campaign 0614c — hypothesis battery on the HS-JEPA wall

> Session goal (user): "제출 csv는 만들지 말고, 가능한 많은 가설을 세워 검증하며 성능 체크."
> 15 offline hypothesis tests run against the authoritative `polytope_eval` arbiter + CV harness.
> **No submission CSV produced** (per instruction). Tools: `outputs/track*.py`.

## TL;DR
The campaign **independently confirms the wall** AND **sharpens its mechanism** with several
findings the prior handoff did not state (or stated less precisely):

1. **Certification is JOINT, not per-cell** — 0/70 single-cell moves certify. The overshoot's
   certified worst-case comes only from its joint Q2/Q3 direction. (`trackA`, `trackA3b`)
2. **The entire certification rests on ONE measurement** — leave-one-out: dropping the 0.5619
   anchor flips overshoot×0.8 worst-loose −0.00171 → **+0.01777** (cert collapses). Dropping any
   of the other 29 measurements has **zero** effect. So overshoot×0.8 is "scale the single
   validated gamble back 20%," a low-risk interpolation — **not** an independent certification.
   The handoff's "7σ robust" framing over-states it. (`trackFG`)
3. **post-mean is a MIRAGE metric** (operationalizes §4B, demonstrated cleanly) — "move toward
   posterior-mean rate r_hat" gives post-mean −0.005…−0.04 for **every** target group incl. S,
   yet worst-loose is strongly **positive** for all (even Q2Q3). Only **worst-loose** is honest.
   (`trackA3b`)
4. **The overshoot works because Q2/Q3 DRIFT** — temporal split-half self-corr **Q2=0.013,
   Q3=0.093** (huge drift) vs random-split 0.65/0.52; S targets **0.62–0.93** (stable traits).
   forward-CV recency captures the Q2/Q3 drift **direction** (corr with anchor-implied true rate
   **+0.86 / +0.75**) but undershoots magnitude → the overshoot pushes further and wins.
   A deeper "why" than measurement-coverage. (`trackJ`, `trackH`)
5. **S is dry — but as a bounded coin-flip, not provably zero** — S are stable traits ⇒ train
   already calibrates them ⇒ FS near-optimal on S. An S-probe has a *small real* potential
   (−0.002/target if forward-CV direction holds) but the direction is **unverified for S**
   (only Q2/Q3 verified), **uncertified**, with large worst-case downside (**+0.02…+0.07**).
   Confirms §7B as a genuine bounded gamble, now quantified on both sides. (`trackC`, `trackK`)
6. **Q→S transfer absent** — Q-rate→S-rate LOO-R² negative for S1/S2/S3. The apparent S4 lead
   (LOO-R² +0.27) is a 10-subject **noise artifact**: Q3→S4 corr −0.35 collapses to −0.15 on
   dropping one subject, and Q3 per-subject rate is not even self-stable (0.09). (`trackH`,`trackI`)
7. **No new certified lever exists** — complete scan of global per-target constant shifts
   certifies **none** (incl. Q1-up); the decorrelated ensemble-blend level component **hurts**
   (worst-loose +0.006…+0.016, 64% of its move is non-transferable residual). Only the
   per-subject Q2/Q3 overshoot certifies. (`trackL`, `trackD`)
8. **σ recommendation confirmed & refined** — independent σ-sweep: post-mean optimum σ*≈0.75–0.80
   (handoff said 0.8 ✓); uniform σ **dominates** separate (σ_Q2,σ_Q3) on a risk-adjusted basis;
   lower σ (0.5–0.7) is safer on worst-case; conclusion is **robust to ±0.0006** anchor noise
   (σ*∈[0.70,0.75], sign stays certified). (`trackFG`, `trackE`)

## Decision implications (for when a submission is requested)
- **Best non-gambling move stays overshoot×0.8** (or slightly more conservative **×0.7** for a
  better worst-case). Expected gain over the 0.5619 anchor is **small** (~−0.0005 post-mean /
  ~−0.0017 worst-case) and entirely anchor-dependent. It is a safe refinement, not a breakthrough.
- **The only path to further PUBLIC gain is spending an LB slot on an S-probe** — a bounded
  coin-flip: upside ~−0.003…−0.006 if the S-block drift direction matches forward-CV (unverifiable
  offline), downside bounded but real. S2 or S3 is the highest-information single probe
  (most stable trait × largest spread, and S2–S3 corr 0.86 ⇒ one probe informs the block).
- **Private LB**: per-subject-uniform moves are assignment-noise-free ⇒ public gain ≈ private gain
  (the realized 99.3% transfer). So overshoot×0.8 is private-safe; the residual private risk is
  the single-measurement noise, which `trackE` shows does not flip the sign.

9. **FS calibration is already certified-optimal** — every per-subject recalibration variant
   (toward train rate / last-obs / de-shrink / shrink), all target groups, gives **positive**
   worst-loose (+0.006…+0.47). Only the anchor-measured Q2/Q3 overshoot improves FS. (`trackM`)

## What is now exhaustively closed (do not re-test)
- Single-cell certification, per-cell box-tightening, toward-r_hat harvest (all targets),
  toward-forward-CV / toward-train / toward-last-obs / de-shrink / shrink recalibration,
  separate per-target σ, global constant shifts (all 7 targets), decorrelated ensemble blend,
  Q→S cross-target inference (incl. the S4 mirage), Q1-up as a second lever.
- **Net: the only certified lever is the per-subject Q2/Q3 overshoot, scaled ≈0.7–0.8.
  No offline path to a NEW certified gain exists. Further public gain ⇒ an LB S-probe slot.**

---

# Part 2 — Public-subset identification + multi-S-probe OED (follow-up: "how to beat 0.5619")

## TL;DR (Part 2)
**On the PRIVATE LB (the prize), 0.5619-class overshoot is at/near the achievable floor.**
The only private-positive lever (Q2/Q3 *temporal drift*) is already harvested. Every remaining
"available" public gain (S, Q1) lies **below a drift threshold (~0.12)** and is therefore
**public sampling-noise fitting that HELPS public but HURTS private.** So lowering the *public*
score below ~0.5619 via new axes is possible but **private-negative**. Do not pursue it.

## N. Public-subset identification (`trackN`)
- The 30 LB measurements identify **9 effective dimensions** of the 70-dim per-subject public
  rate vector (SVD of tolerance-normalized measurement matrix; top dirs = Q3 sv49, Q2 sv38).
- **S is NOT fully unmeasured** (handoff overstated): dir2 (sv 11.7) is 30% S1 + 33% S2; per-target
  identified fraction S2=0.15 (> Q1=0.12), S1=0.10. But identification pins r_S ≈ FS's value
  (no gain) and is too weak to certify a cell move.
- **Drift vs composition decomposition**: Q2 public gap (0.19) = 73% forward-CV drift + 0.107
  composition residual; Q3 (0.14) = 56% drift + 0.108 residual. The composition residuals
  correlate across Q2,Q3 at **+0.43** — a *weak* shared public-composition factor exists, but it
  is moderate and (being a within-subject night-selection effect) transfers poorly to *stable* S.

## O. S-block OED ceiling (`trackO`) — and why the naive number is a mirage
- Using the polytope posterior (box ±0.45) as prior, the "infinite-probe S ceiling" computes to
  −0.045 and even "calibrate-to-mean, no probe" to −0.021 with p(improve)=1.0. **This is the SAME
  over-anchor mirage** (the box prior is unrealistically wide); trackC/M already showed these
  S moves have positive worst-loose. Not real. Motivates the realistic-prior analysis (P).
- S-rate uncertainty is genuinely low-rank: **5 probe directions remove 80%** of S variance
  (S2–S3 corr 0.86), so few probes *could* pin the block — but pinning buys nothing real (see P).

## P. The decisive test — private transfer of calibration (`trackP`)
Realistic structure: **~12 public rows/subject** ⇒ per-subject public rate SE ≈ 0.14 (huge).
Calibrating to the public rate fits that noise; it transfers to PRIVATE only if it corrects a
SHARED signal (temporal drift in both halves), not public-only sampling noise. Monte Carlo
(n_train45/n_pub12/n_priv12), perfect public calibration, gain vs drift:

| drift | E[public gain] | E[private gain] |
|------:|---------------:|----------------:|
| 0.00  | −0.056         | **+0.038** (hurts private) |
| 0.10  | −0.079         | +0.012 |
| 0.12  | −0.090         | **−0.002** (threshold) |
| 0.19  | −0.143         | −0.064 |

- **Private gain crosses 0 at drift ≈ 0.12** = the public sampling-noise floor √(σ²_pub−σ²_train).
- **Q2 (drift 0.19) / Q3 (0.14): ABOVE threshold ⇒ overshoot helps private** — why 0.5619 won.
- **S1–4 (systematic drift ≈ 0.02–0.03, from split-half self-corr 0.62–0.93): far BELOW ⇒ an
  S-probe gains PUBLIC (a lot) but LOSES PRIVATE.** S-probing is **private-negative-EV**.
- Note the large public gains (−0.05…−0.19) are exactly the §7A public-overfit trap: tempting on
  public, toxic on private. The metric that ranks the paper competition is PRIVATE.

## Multi-S-probe OED sequencing — design + verdict
- *If one only cared about public*: spend ≈5 probes along the top S-eigendirections (S2/S3 first,
  they share 86% corr; then S1, S4), each a per-subject S-level submission; pin 80% of S variance,
  then submit the calibrated S-move. Public could drop well below 0.5619.
- **Verdict: do NOT run it.** Every S-probe calibrates to public sampling noise (S drift < 0.12
  threshold) ⇒ each unit of public gain is bought with private loss. The sequencing is sound OED
  but optimizes the wrong objective. Reserve slots; keep the private-safe overshoot.

## Final answer to "lower 0.5619"
- **Private LB**: 0.5619 (overshoot) ≈ the floor for this base; **overshoot×0.8 (~0.5617) is the
  only private-safe refinement** (same drift axis, mild winner's-curse shrink). No other offline
  lever is private-positive — Q1 and all S are stable (sub-threshold drift), per-row signal is
  exhausted (handoff), ensemble/recalibration don't certify.
- **Public LB**: can be pushed lower via S/Q1 noise-calibration, but that **raises private**. Only
  pursue if the goal were public-leaderboard optics, not the prize — explicitly not recommended.

---

# Part 3 — three follow-ups: S-drift validation, per-row re-attempt, private-rank robustness

## (a) S-drift estimate, rigorously (`trackQ`, `trackQ2`) — verdict survives, sharpened
- Split-half |late−early| is noise-inflated; the honest signal is the permutation-tested trend +
  block variance decomposition. **Significant systematic drift: Q2 (p=0.000), Q3 (0.022),
  S2 (0.032), S4 (0.066 marginal).** S1/S3/Q1 are pure noise. So S is **not** drift-free — S2/S4 drift.
- But the magnitude matters: noise-subtracted drift SD = Q2 0.151, Q3 0.117, **S2 0.088, S4 0.078**,
  S1 0.064, S3 0.038. Feeding each into the trackP private-transfer Monte Carlo (real n_pub~12):
  **E[private gain] of a probe-informed S calibration is POSITIVE (hurts) for every S target**
  (S1 +0.028, S2 +0.020, S3 +0.035, S4 +0.024) — because even S2's real drift (0.088) sits
  **below the ~12-row public sampling-noise floor (~0.14)**. Only Q2 (−0.017) clearly clears it;
  Q3 (+0.005) is borderline. ⇒ **the private-negative S-probe verdict is robust to the S2-drift
  objection.** Confidence now HIGH.

## (b) Per-row signal re-attempt, drift-aligned (`trackR`) — DEAD
- The strongest defensible per-row hypothesis: a per-row LINEAR-TRAJECTORY shift (each row moved
  toward its time-extrapolated rate) — per-row but drift-aligned, so it *should* transfer.
- Test-faithful CV, future vs interleaved: the trajectory is **far WORSE** than the uniform
  per-subject recency shift on the drifted FUTURE rows (Q2 traj−uni **+0.105**, Q3 **+0.153**).
  The per-subject slope (~38 train rows) is noise; extrapolating it per-row amplifies that noise.
- ⇒ **The uniform per-subject shift already captures all transferable drift; per-row refinement
  only adds non-transferable estimation noise.** Combined with the handoff's exhausted encoders
  (ROCKET/CNN/JEPA/LLM), **per-row is conclusively dry** — even the drift-aligned angle fails.

## (c) Private-rank robustness + final-submission hedge (`trackS2`, polytope-faithful)
- For the per-subject-uniform overshoot, **E[private gain] ≈ E[public gain] at every σ**
  (the transfer property), private E-optimal **σ=0.8 (−0.00527)**. Risk-return: lower σ is safer
  (σ=0.4 → P(improve)=0.95, CVaR5 +0.001; σ=1.0 → P=0.79, CVaR5 +0.007).
- **DACON keeps the better of 2 final submissions on private ⇒ a 2-scale hedge dominates a single σ:**

  | final pair | E[best-of-2] | P(best<0) | CVaR5 |
  |---|---:|---:|---:|
  | x0.8 alone | −0.00527 | 0.86 | +0.0044 |
  | **{x0.7, x1.0}** | **−0.00794** | 0.97 | +0.0004 |
  | {x0.6, x1.0} | −0.00771 | 0.98 | +0.0000 |
  | {FS, x0.8} | −0.00560 | 0.86 | **0.0000 (downside capped)** |

  Bracketing the scale ({x0.7,x1.0}) captures ~50% more expected private gain than a single σ and
  nearly removes the downside tail; {FS, x_overshoot} guarantees ≥ FS on private (a free option).
- **Recommendation for the eventual final pick** (no submission made yet): submit two —
  **overshoot×1.0 (max expected) + overshoot×0.6–0.7 (scale hedge)**, OR if maximal safety is
  wanted, **overshoot×0.8 + FS** (capped downside). Either beats a lone σ on the private metric.

## Tooling (published in `verification_campaign/`)
`trackA_cell_certification`, `trackA2_measurement_coverage`,
`trackA3b_toward_rhat`, `trackC_probe_gamble`, `trackD_ensemble_blend`, `trackE_measurement_noise`,
`trackFG_sigma_and_loo`, `trackH_predictor_reliability`, `trackI_cross_target_s4`,
`trackJ_trait_stability`, `trackK_drift_drives_gain`, `trackL_global_shift_scan`,
`trackM_fs_recalibration`, `trackN_public_identification`, `trackO_sprobe_oed`,
`trackP_private_transfer_sim`, `trackQ_s_drift_validation`, `trackQ2_per_target_private_ev`,
`trackR_perrow_trajectory`, `trackS2_private_robust_polytope`,
plus `save_overshoot_x08.py` (new-best builder) + `build_certified_dry.py` (upload-safety dry run).
All reuse `polytope_eval.py` (authoritative arbiter) + (for `trackR`) the team-workspace `hsjepa_core.py`
CV harness + competition train labels. Two scratch iterations were superseded and omitted:
`trackA3` (joint-cellset optimizer that failed to descend from d=0 → replaced by `trackA3b`) and
`trackS` (a generative private-robustness model with a calibration artifact → replaced by `trackS2`).
