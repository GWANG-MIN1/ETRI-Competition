# HS-JEPA paper-direction briefing (for workflow agents)

## The competition is a PAPER competition
제5회 ETRI 휴먼이해 AI 논문경진대회 (ETRI "Human Understanding"). The deliverable is a **paper**; the team
will write it from whoever's idea scores best. So the goal is BOTH leaderboard performance AND a genuine,
defensible, novel research contribution about understanding human (sleep/stress) state from lifelog data.

## Data / task
- 10 subjects, 450 train rows / 250 test rows. 7 BINARY targets, scored by mean **logloss** (lower better).
  - Q1,Q2,Q3 = SUBJECTIVE wellbeing ("today vs this person's own average": sleep-quality / fatigue / stress).
  - S1=TST, S2=SE, S3=SOL, S4=WASO = OBJECTIVE NSF sleep-guideline pass/fail (Withings sensor).
- Inputs: smartphone + smartwatch lifelog (HR, activity, screen, light, GPS, BLE, WiFi, ...).
- Test = the SAME 10 subjects as train; 62% interleaved within train dates, 38% future, 0% past.

## What ~50 experiments actually established (the real regime)
- **Calibration-dominated, NOT feature-learning.** Per-night discriminative signal is at the noise floor
  (model AUC ceilings Q2~.71, Q3~.67; sensor-derived features systematically SUBSUMED by the base).
- **Only LEVEL/PRIOR shifts transfer** to the 250-row LB; per-row reranks are sign-unreliable (a mean-preserving
  rerank moved LB the WRONG way). The CV-LB gap is the central methodological hazard.
- Validated levers: per-subject base rate + temporal drift (recency) + public-prior elevation (Q-up).
- New, honest module results (test-faithful CV, 12 seeds): V131C cohort *shrink* (regression-to-mean) is REFUTED;
  cohort *DE-shrink* (amplify personal-vs-cohort gap) HELPS S1/S3; transition/hysteresis level-shifts REFUTED
  (base already absorbs them); S2 global-down robust; Q-up is a public-prior bet (not CV-visible).
- "Listener responsibility" formalized as **transfer-audibility**: each single-target LB probe is an equation for
  the public base rate; back-out gives p*(Q2)~.78, p*(Q3)~.74 (constant-approx, over-estimates magnitude).

## THE LIMITATION TO OVERCOME (user's words)
Everything so far feels like a public-LB-fixated experiment log, not a paper. Risks: (1) public != private
(prior-fitting may not generalize), (2) thin science ("we shifted a mean by 0.03"), (3) no human-understanding insight.

## HS-JEPA framework (team-merged) + components to elevate from tricks -> principles
hidden human-state -> listener responsibility -> toxicity/safety -> row-target correction. Components:
State-Transition Verb, Probability Geometry Guard, Chronotype Phase Warping, LogLoss Regret Allocator,
Event Boundary Detector, Shadow Public Subset Mapper, Proxy Context Expansion, Q-S Lag-Hysteresis, V131C cohort gap.

## Seed directions (BUILD ON or CRITIQUE these; be specific & grounded, no generic ML fluff)
1. **Hierarchical / empirical-Bayes latent state**: model p(target | subject, time) with partial pooling
   (cohort prior) + temporal drift; HS-JEPA "hidden human-state" made rigorous. Subsumes recency/cohort/de-shrink
   as one coherent posterior. Generalizes to leave-subject-out / leave-time-out.
2. **Transfer-audibility as a general diagnostic** (listener responsibility): given a known train->test shift,
   predict which CV gains will transfer; retrospectively explains why rerank failed but level shifts won.
   = a methodology against public-LB overfitting (broadly useful, citable).
3. **Decision-theoretic submission under a proper scoring rule** (Regret Allocator formalized): optimal mean-shift
   given uncertainty about the test distribution; replaces hand-tuned nudges with derived ones.
4. **Joint multi-target manifold consistency** (Geometry Guard): the 7-target joint prediction must lie on the
   plausible human-state manifold; structured/joint calibration.
5. **Human-understanding finding**: SUBJECTIVE (Q) vs OBJECTIVE (S) sleep mismatch — for whom does objective sleep
   predict subjective wellbeing, for whom does it diverge (e.g., id06 sleeps great objectively, rates poorly)?
   Individual differences in sleep perception/interoception. THIS is the "Human Understanding" deliverable.
6. **Latent state-transition dynamics** (State-Transition Verb): per-subject wellbeing as a latent dynamical state
   (recovering/declining); crude level-shift failed, but a proper generative dynamics model is paper-worthy and
   directly on-theme.

## TASK
From your assigned lens, produce a COMPLETE, SPECIFIC, grounded research+paper direction: thesis, method, novelty
vs prior art, an EVALUATION protocol that does NOT depend on public-LB fixation, the human-understanding insight,
how it explicitly overcomes the LB-fixation limitation, whether/how it also helps LB, risks, and a minimal viable version.
