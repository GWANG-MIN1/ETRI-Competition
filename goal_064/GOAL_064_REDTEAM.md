# GOAL_064 — Red-team of "detectable but not correctable" (adversarial)

작성일 2026-06-19 · 목표: 결론을 옹호 말고 무너뜨릴 가설 찾기. 도구 `goal064_redteam.py` + `goal064b_confirm.py`.

## 개념적 wedge (옳음, 부분 인정)
overshoot 자체가 **correctable direction**(subject-level Q recency drift, 방향=위, LB 0.5615 승리)이므로
"not correctable"은 **과장**. 정확한 명제 = "배포된 Q-overshoot 외에 *추가* correctable direction이 있나?"

## 3 메커니즘 falsification
- M1 aggregate(subject-level) 방향 / M2 latent drift trajectory / M3 regime-conditional(상쇄형).
- 1차(loose): M1:S1, M2:Q1/S2/S4 "break"로 보였으나 — **방법론 함정**: future block 작아(~타깃당 수십행)
  랜덤 피처도 \|corr\| 0.15~0.26 냄. 1차 placebo=단일 draw라 우연히 낮음.

## 엄격 확인(goal064b: 30-shuffle placebo NULL 분포 + transfer-gate)
| 후보 | future corr | placebo\|95%\| | 격파 | gate |
|---|---|---|---|---|
| M2 Q1 | +0.106 | 0.156 | ❌ | LOTTERY |
| M2 Q2 | −0.111 | 0.233 | ❌ | LOTTERY |
| M2 S2 | +0.165 | 0.256 | ❌ | LOTTERY |
| M2 S4 | +0.195 | 0.260 | ❌ | NO_GAIN |
| M1 S1 | — | — | — | LOTTERY |
| M3 7/7 | 0.13~0.22 | 0.20~0.45 | ❌ 전부 | — |
**모든 break 소멸**(placebo null 내 + gate LOTTERY/NO_GAIN). PASS_LEVEL·유의미 DATE_BOUND 0.

## 결론 (적대적 검증 후 — 강화됨)
- 정확한 재명제: **"correctable direction은 deployed Q-recency overshoot 단 하나뿐; subject-level/latent-drift/
  regime-conditional 어느 쪽도 *추가* 방향 신호 없음(엄격 placebo+gate). overshoot 너머는 detectable(크기) but
  not correctable(방향) 유지."**
- M3(conditional-miscalibration 대안)도 placebo 못 넘어 기각 → 가장 단순한 가설 유지:
  **base가 관측·latent 피처에 well-calibrated → E[r|feat]≈0(방향0), Var만 예측가능(GOAL_062).**
- 세션 메타교훈: 작은 future block corr 노이즈 + 다중비교 → loose 임계 위양성 **3회 반복**(GOAL_059/063/064).
  **신뢰 판별자 = proper placebo NULL 분포 + transfer-gate.** (논문 방법론 핵심.)

8축 소진. 0.54 경로 없음, 정보바닥 ≈0.56, anchor 0.5615 best. CSV 0.
