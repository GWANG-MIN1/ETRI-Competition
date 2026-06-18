# GOAL_063 — Can we predict base error DIRECTION (signed residual), not just magnitude?

작성일 2026-06-19 · GOAL_062 후속(크기는 예측됐으나 방향 미지로 보정 실패). 도구 `outputs/goal063_error_direction.py`, 확인 `goal063b_confirm.py`.

## 사전 사실
이진 타깃이라 `sign(y−p_base) ≡ y` → "방향 분류"는 문자 그대로 라벨 예측. 의미있는 질문 = **관측 피처(결측+리듬+subject-상대+HS-JEPA)가 base가 못 잡은 signed residual `y−p_base`를 예측하는가**(=방향성 보정 가능?). residual 회귀로 검증(held/FUTURE/LOSO + placebo null + 다중비교 점검).

## 결과
1차 스윕(loose): Q2·S3가 임계 넘음 → **엄격 확인(seed 부호 + 30-shuffle placebo null 95%pt)**:
| 후보 | future corr | pos seeds | placebo 95%pt | 판정 |
|---|---|---|---|---|
| S3 risk-only | +0.126 | 9/10 | +0.119 | 간신히 통과 |
| S3 risk+z | −0.061 | 3/10 | +0.114 | FAIL(z서 소멸) |
| Q2 risk+z | +0.081 | 7/10 | +0.183 | FAIL(null 내) |
| Q2 risk-only | −0.023 | 4/10 | +0.178 | FAIL |
- sign-AUC(feat→y) 전 타깃 0.45~0.57 ≪ base AUC 0.60~0.77 = base 넘는 방향정보 없음.

## 결론 = FAIL (방향 robust 예측 불가)
- 6/7 타깃 NO. **S3(SOL)만 단일 엄격테스트 간신히 통과**(루틴+결측→onset 편향 가능성)이나 ① 7타깃 **다중비교 미보정**(Bonferroni ~99%pt 필요→탈락) ② **z 추가 시 소멸**(fragile) ③ GOAL_062 gate서 S3 submargin → **actionable 아님**.
- 근본 이유: **base가 관측 피처에 well-calibrated**(피처가 base 5242에 포함)→`E[y−p|feat]≈0`(방향=0), `Var`만 예측 가능. GOAL_062(크기 예측)+GOAL_063(방향 불가)=통계적으로 일관.
- 이론적 상한: S3 hint 액면가라도 future S3 ~−0.008 → 7t평균 ~−0.001, 전이 불확실. **0.54 아님.**
```
GOAL_063_RESULT = FAIL (direction not robustly predictable; magnitude-only, confirms GOAL_062)
CSV = 0. anchor 0.5615 유지.
PAPER: "base error magnitude is predictable from missingness/routine (GOAL_062) but its SIGN is not
       (base well-calibrated wrt those features) — so detection cannot become correction." 깔끔한 정리.
```
7축(Q overshoot/S 센서천장/ensemble/external Walch/cohort/error-magnitude/error-direction) 소진. 정보바닥 ≈0.56.
