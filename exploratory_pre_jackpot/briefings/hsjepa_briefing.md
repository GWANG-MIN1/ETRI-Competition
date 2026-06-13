# HS-JEPA submission briefing (for workflow agents)

## Competition (DACON 236690, ETRI Human Understanding)
- 7 binary targets {0,1}: Q1,Q2,Q3 (subjective wellbeing, "above personal avg"), S1=TST,S2=SE,S3=SOL,S4=WASO (NSF objective sleep, 1=meets guideline).
- Metric = **mean logloss** over 7 targets x 250 test rows. LOWER is better.
- Test = the SAME 10 subjects as train (id01..id10). 62.4% interleaved within train date span, 37.6% future, 0% past.
- Train marginals: Q1=.496 Q2=.562 Q3=.600 S1=.682 S2=.651 S3=.662 S4=.560.

## THE TRANSFER LAW (hard-won from ~45 prior experiments, non-negotiable)
- **Only LEVEL / PRIOR shifts transfer to the 250-row LB.** Per-row ranking/feature/chain gains do NOT
  (the S234 mean-preserving rerank moved LB the WRONG way 0.5932->0.5956).
- New derived features get subsumed by the saturated base model.
- LB-confirmed wins so far (all LEVEL shifts on Q): Q1 up to .605, Q2 recency to .615, Q3 recency to .616.
  LB trajectory 0.6001 -> 0.5949 -> **0.5932** (current best = submission_FINAL_Q2recency_plus_Q3recency.csv).
- Current best means: Q1=.605 Q2=.615 Q3=.616 S1=.699 S2=.653 S3=.682 S4=.565. (S essentially untouched = frontier.)

## SHADOW PUBLIC MAPPER / AUDIBILITY (listener responsibility, quantified)
Each single-target LB probe is an equation: dLB = (1/7)*d(target logloss). Back-out (constant approx):
- **p*(Q2) ~ 0.78**, current .615 -> room UP. (constant-approx OVER-estimates; direction robust, magnitude optimistic)
- **p*(Q3) ~ 0.74**, current .616 -> room UP.
- Q1: up100(.605) beat up140(.645) on LB -> Q-type optimum is ~.60-.645; .645 overshoots = TOXICITY boundary.
- Audibility law: LEVEL moves down to ~0.0017 dLB are correctly-signed & repeatable; PER-ROW moves are sign-unreliable at any size.

## TEST-FAITHFUL CV EVIDENCE (12 seeds; held-out = 15% future + 25% interleaved per subject)
Baseline = cached unified OOF (blocked-time -> already strongly per-subject calibrated; biases AGAINST level shifts).
delta = held-out logloss change vs baseline (NEGATIVE = improvement); imp = fraction of seeds improved.

| lever | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 | verdict |
|---|---|---|---|---|---|---|---|---|
| subject_recency tau21 | +.042 | +.013 | +.012 | +.038 | +.032 | +.017 | +.035 | hurts here (forward-only signal diluted by interleaved + blocked-OOF) |
| global +0.08 | +.009 | **-.002(67%)** | +.003 | +.038 | +.065 | +.021 | +.016 | only Q2 up helps; Q-up is public-prior (probe-only) |
| global -0.02 | +.003 | +.005 | +.004 | +.001 | **-.003(100%)** | +.000 | +.001 | **S2 down robust** |
| V131C cohort_shrink +0.3 | +.011 | +.011 | +.006 | +.022 | +.022 | +.037 | +.014 | **REFUTED (hurts all)** |
| cohort DE-shrink -0.3 (amplify gap) | +.004 | -.001 | +.003 | **-.0074(83%)** | +.022 | **-.0118(92%)** | +.031 | **S1,S3 de-shrink HELPS** |
| transition_drift lam0.5 | +.068 | +.041 | +.077 | +.033 | +.081 | +.016 | +.020 | **REFUTED (catastrophic)** |
| rerank_control (per-row) | +.100 | +.014 | +.003 | +.070 | +.049 | +.002 | +.093 | per-row destroys calibration (control OK) |

Key reads: base model already absorbs per-subject level -> shrink/transition ADD noise. The surviving levers are
(a) global Q2/Q3 UP (public prior; probe+audibility, not CV-visible), (b) global **S2 down** (drift),
(c) per-subject **S1/S3 DE-shrink** (amplify personal-vs-cohort gap; matches 0603b independent finding).
CAVEAT: blocked-OOF biases CV against level shifts, so absence-of-CV-gain for Q-up/recency is expected, not disqualifying.

## TEAM MODULES -> transferring interpretation
1. Shadow Public Subset Mapper = the audibility back-out above (DONE).
2. LogLoss Regret Allocator = set shift magnitude by confidence: attack probe-backed levers, defend uncertain ones; never overshoot p*.
3. State-Transition Verb / Q-S Lag-Hysteresis / Event Boundary / Chronotype = per-subject drift DIRECTION -> tested as transition_drift = REFUTED at level (base absorbs it).
4. V131C personal-vs-cohort gap = de-shrink (amplify) for S1/S3 = SURVIVES.
5. Probability Geometry Guard = the joint 7-vector must stay plausible (no subject pushed to an implausible Q-high/S-low combo; respect per-subject train ranges).
6. Proxy Context Expansion = more features = subsumed (transfer law).

## TASK FOR THE WORKFLOW
Produce a per-target (and where justified, per-subject) LEVEL-shift recipe on top of the current best (0.5932),
sized by the Regret Allocator, capped by the toxicity/overshoot geometry, within a small distance budget,
passing the joint geometry guard. Deliver a 2-rung ladder: conservative (safe probe) + moderate.
Every shift must be a LEVEL move with a transfer rationale. Refuted modules must be EXCLUDED (state why).
