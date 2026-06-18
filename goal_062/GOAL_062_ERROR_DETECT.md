# GOAL_062 — Base-error detection via missingness + life-rhythm → selective calibration

작성일 2026-06-19 · 프레임: 수면 예측이 아니라 "누가/언제 base가 confident-wrong인지"를 관측가능
(결측+생활리듬, 라벨 불필요)한 신호로 탐지 → 고위험 행만 안전 prior로 후퇴 → confident-wrong logloss 절감.
도구 `outputs/goal062_error_detect.py`. 위험 피처는 canvas obs(결측)/X(루틴)에서 추출(30개).

## Phase 1 (GO/NO-GO) = PASS ⭐ (이번 캠페인 유일의 OOS placebo-격파 신호)
- risk 피처 → base row_loss 예측 Spearman: held-out **+0.164**, **FUTURE +0.187**, placebo **+0.002** (마진 +0.16).
- 즉 **base 불신뢰 행은 미래에도 위약을 명확히 이기며 예측 가능.** 핵심: nm_screen_on(0.215), dev_active_rate, nm_light, cov_motion/step/speed(결측). q_loss는 약함(future +0.135, 마진 +0.03).

## Phase 2-3 (보정 + 검증) = FAIL
- 고위험 행 → subject-mean shrink:
  - Q 도움(q0.33,w0.5: Q −0.0048) 그러나 **risk-gate가 random-gate 못 이김**(risk<random False 전 조건) = 탐지 특이성 없는 일반 shrinkage.
  - S 악화(+0.0020~+0.0046). 순 7t ≈ 중립/약간 악화.
  - transfer-gate: Q1/Q3/S3 submargin, Q2 LOTTERY, 나머지 NO_GAIN → **PASS_LEVEL NONE**.

## 핵심 결론 (논문 가치 있는 발견)
**"base 오류는 결측·생활리듬으로 예측 가능(OOS·placebo격파)하지만, 교정 가능한 방향이 없어 logloss로 전환 불가."**
1. 오류 크기는 예측되나 **방향 미지** → 안전한 수는 prior 후퇴뿐.
2. shrink-to-prior는 **일반 효과**(random-gate 동급) → 탐지 특이성이 이득화 안 됨.
3. Q 후퇴는 LB 승리 레버 overshoot(위)와 **반대 방향** → CV 게인도 LB 전이 안 됨. S 후퇴는 악화.

## 결론
```
GOAL_062_RESULT = FAIL (detectable but not correctable)
CSV = 0 (predictive=True, but gate PASS_LEVEL=NONE & risk-gate not > random-gate)
BEST = anchor submission_overshoot_x0p8_330ef1a1 (0.5615333471) 유지
PAPER VALUE = HIGH: 관측 신호로 base 불신뢰 예측 가능(Spearman~0.19 future)을 실증 — 단 교정 불가 이유까지 규명.
```
6축(Q overshoot / S 센서천장 / ensemble CV-LB / external Walch / cohort / error-detect) 전부 소진. 정보바닥 ≈0.56.
