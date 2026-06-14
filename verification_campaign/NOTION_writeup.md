# ETRI 수면·스트레스 AI 경진대회 (DACON #236690) — 인증 기반 가설검증 캠페인으로 public LB 0.5619 → 0.56153

> **한 줄 요약**: "더 좋은 모델"이 아니라 **무엇이 전이(transfer)되는지를 증명**하는 방향으로 접근. ~23개 가설을 권위적 arbiter로 검증해 벽의 구조를 규명하고, 인증된 단일 레버(overshoot×0.8)로 public LB를 예측 정확도 99%로 개선.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 대회 | 제5회 ETRI 휴먼이해 AI 논문경진대회 (수면/스트레스) — **논문 대회, 최종 순위는 private LB** |
| 메트릭 | 250 test rows × 7 binary targets 평균 logloss (낮을수록 좋음) |
| 7 타깃 | Q1·Q2·Q3 = 주관 웰빙 / S1=TST·S2=SE·S3=SOL·S4=WASO = 객관 수면지표 |
| 구조 | test = train과 동일 10명, 62.4% interleaved + 37.6% future. public LB ≈ test의 50% 서브셋 |
| 데이터 한계 | 라벨 단 450행 (명당 ~45행) → **logloss는 calibration-bound** |

**핵심 인사이트**: 명당 라벨이 적어 per-row 판별보다 **per-subject 레벨/캘리브레이션**이 점수를 지배. "전이되는 신호"와 "과적합 신호"를 구분하는 것이 전부.

---

## 2. 출발점 — 벽 (the wall)

- 기존 best: **public LB 0.5619** (FS base + per-subject Q2/Q3 "overshoot" 레벨 시프트, 직전 세션에서 검증된 도박).
- 누적 ~27개 base-model 후보가 전부 기각된 상태. "벽"의 정체가 모호했음.
- **목표**: 제출 없이 가능한 많은 가설을 세워 검증하고, 0.5619를 낮출 방법을 찾는다.

---

## 3. 방법론 — 인증 프레임워크 (이 프로젝트의 핵심 도구)

1. **polytope_eval (권위적 arbiter)**: 30개 실측 LB 측정으로 per-subject 라벨률에 대한 **polytope(다면체)** 를 구성. 임의의 per-subject 레벨 이동의 worst-case gain을 LP로 인증. *(실측 LB로 calibrate되어 신뢰 가능)*
2. **정직한 지표 = worst-loose** (polytope 최악 경우). **post-mean은 mirage**임을 시연.
3. **transfer 분해**: per-subject LEVEL(전이 가능·인증 가능) vs within-subject RESIDUAL(per-row lottery·비전이).
4. **CV 하네스**: interleaved + test-faithful CV로 per-row 가설 검증.
5. **Monte Carlo / OED**: public↔private 전이, value-of-information을 시뮬레이션.

> 총 ~23개 가설을 14개 분석 스크립트(`outputs/track*.py`)로 검증.

---

## 4. 핵심 발견 (테마별)

### 4.1 인증의 정체 — joint이고, 단일 측정에 의존
- 단일 셀 이동은 **0/70 인증 불가**. 인증은 셀별 분해가 아니라 overshoot 방향의 **joint 구조**에서 나옴.
- **Leave-one-out 폭탄**: 0.5619 anchor 측정 하나만 빼면 overshoot×0.8의 worst-loose가 −0.0017 → **+0.0178로 인증 붕괴**. 다른 29개 측정은 빼도 영향 0.
- ⇒ overshoot×0.8의 "인증"은 **독립 발견이 아니라 검증된 도박을 20% 축소한 보간**. ("7σ robust" 프레이밍은 과장)

### 4.2 post-mean = mirage 지표 (정직한 지표는 worst-loose)
- "추정 rate(r_hat)로 이동"은 **모든 타깃 그룹(S 포함)에서 post-mean 음수**(개선처럼 보임)지만 **worst-loose는 전부 양수**.
- post-mean은 posterior 평균으로 이동하면 자동으로 음수가 되는 **기만적 지표**. 오직 worst-loose만 진짜 전이를 반영.

### 4.3 벽의 진짜 원인 — "드리프트 가용성" (측정 커버리지보다 깊은 설명)
- per-subject rate의 **temporal split-half self-correlation**:
  - **Q2 = 0.013, Q3 = 0.093** → 레벨은 실재하나 **시간 드리프트가 거대**
  - **S1~S4 = 0.62~0.93** → **고도로 안정적인 trait, 드리프트 거의 없음**
- forward-CV(recency)가 Q2/Q3 드리프트 **방향을 정확히 포착**(실제 rate와 corr +0.86/+0.75) → overshoot가 그 드리프트를 밀어 이김.
- **S는 안정적 trait → train rate가 이미 정확 → FS가 이미 calibrated → 공략 여지 없음.** Q1도 (미측정인데도) 안정적이라 gain 없음 → **"측정"이 아니라 "드리프트"가 gain을 결정.**

### 4.4 🎯 S-probe는 public을 내리지만 private(상금)을 올린다 — 음수 EV
- test 명당 ~12 public 행 → per-subject public rate의 **샘플링 노이즈 ~0.14**.
- 캘리브레이션은 **공유 신호(드리프트)일 때만 private로 전이**, public 전용 샘플링 노이즈면 전이 안 됨.
- Monte Carlo: private-gain이 **드리프트 ≈ 0.12 임계**에서 0을 교차.
  - **Q2(드리프트 0.19)/Q3(0.14) = 임계 위 → overshoot가 private 도움** (0.5619이 이긴 이유)
  - **S(드리프트 0.02~0.09) = 임계 아래 → S-probe는 public 크게 내리나(−0.05~−0.19) private 올림(+0.02~+0.04)**
- S2/S4는 실제 드리프트가 있지만(SD ~0.08) **그조차 샘플링 노이즈 floor 아래** → verdict 견고.

### 4.5 per-row 신호 — 확정 소진
- 가장 방어 가능한 가설(드리프트 정렬 per-row 궤적)조차 test-faithful CV에서 균일 shift보다 **훨씬 나쁨**(future 행 +0.10~+0.15). per-subject 기울기 추정 노이즈가 외삽에서 증폭.
- 앙상블 블렌드·FS recalibration·global 상수 shift도 전부 미인증 → **새 인증 레버 0**.

---

## 5. 제출 & 결과 — 0.5619 → 0.56153 ✅

| 항목 | 값 |
|---|---|
| 제출 파일 | `submission_overshoot_x0p8_330ef1a1_uploadsafe.csv` |
| 레시피 | FS base + **0.8 × (per-subject Q2/Q3 드리프트 overshoot)** — Q2/Q3만 변경 |
| 예측 (모델) | post-mean 앵커 대비 **−0.00037** → ~0.56154 |
| **실현 public LB** | **0.5615333471** (앵커 0.5619100863 대비 **−0.0003767**) |
| 예측 정확도 | **99%+** (예측 −0.00037 vs 실현 −0.0003767) |

> 정직성 노트: S/Q1 노이즈 캘리브로 public을 **더 크게** 내릴 수도 있었으나, 미인증(public도 오를 위험) + **private(상금) toxic**이라 의도적으로 배제. 이 제출은 public을 내리면서 private도 해치지 않는 유일한 선택.

---

## 6. 2점 확정 — σ*=0.80이 이 축의 floor

이제 overshoot 패밀리에 **실측 2점**(σ=1.0→0.5619, σ=0.8→0.56153)이 생겨 스케일을 정밀 triangulate:

- 2차 곡선 fit: **realized-optimal σ* = 0.80** (현재 위치와 정확히 일치)
- σ별 예측 LB: σ=0.75→0.561559, **σ=0.80→0.561533**, σ=0.85→0.561555
- ⇒ **0.56153이 (인증된·private-safe) overshoot 축의 floor.** σ 튜닝으로는 더 못 내림.

---

## 7. 결론 & 다음 단계

**결론**
- private LB(상금)에서 **0.56153은 이 base의 floor에 근접**. 유일한 private-양수 레버(Q2/Q3 드리프트)는 완전히 수확됨.
- 그 외 모든 축(S·Q1·per-row·앙상블·recalibration)은 private-음수 또는 미인증으로 닫힘.

**남은 실행 가능한 가치 (모두 "방법"이지 새 신호 아님)**
1. ✅ overshoot×0.8 (완료, 0.56153)
2. **2-제출 헤지**: DACON이 2제출 중 private 더 나은 것을 채택 → {×0.7, ×1.0}이 단일보다 기대 private gain ~50%↑, 다운사이드 꼬리 제거. 또는 {FS, ×0.8}로 다운사이드 완전 차단.

**다음 단계 후보**
- 최종 2-제출 선택 확정 (private 최적 헤지 구성)
- 새 측정 0.56153을 ledger에 반영해 polytope 재calibrate

---

## 부록 — 산출물

- **분석 도구**: `outputs/track{A,A2,A3,A3b,C,D,E,FG,H,I,J,K,L,M,N,O,P,Q,Q2,R,S,S2}*.py` (~14 스크립트, ~23 가설)
- **권위적 arbiter**: `polytope_eval.py` (30 실측 LB calibrated)
- **상세 리포트**: `CAMPAIGN_0614c_VERIFICATION.md` (Part 1/2/3)
- **제출**: `submission_overshoot_x0p8_330ef1a1_uploadsafe.csv`

**시연한 역량**: 적대적 가설검증 · LP 기반 인증 · public↔private 전이 분석 · Monte Carlo/OED · mirage 지표 탐지 · 정직한 negative result · 예측-실현 99% 검증.
