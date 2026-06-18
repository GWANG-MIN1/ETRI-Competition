# GOAL_056 — LAST-SHOT aggressive rank-rescue (target-specific Q2/Q3 two-bucket)

작성일 2026-06-18 · 마지막 제출권 1장 · 목표: C1보다 큰 public 개선 가능성의 aggressive 후보 1개 확정
규칙: 새 모델 X, 누수 X, anchor + PE(polytope_eval) 기반만, Q2/Q3 two-bucket overshoot 최적화만, 최종 CSV 1개.

## 방법
- anchor = `submission_overshoot_x0p8_330ef1a1_uploadsafe.csv` (LB 0.5615333471) = FS base + 0.8×Q2/Q3
  recency logit-step. 동일 formulation에 **4개 독립 sigma**: σ_Q2_int, σ_Q2_fut, σ_Q3_int, σ_Q3_fut.
- bucket = `행 sleep_date > subject train-max date`(관측값, 94 future/156 inter = 0.376/0.624).
- grid: int{0,0.2,0.4,0.6,0.8} × fut{0.8,1.0,1.2,1.4,1.6} per target = 25×25 = **625 조합 전수**.
- **PE post-mean은 타깃별 정확 가법**(gain = per-(subject,target) 합 — 증명·검증됨 1e-9) → 타깃별 25회만 sweep 후 합산.
- worst-loose(joint, 비가법)는 shortlist(상위60+named)에 대해 loose polytope linprog로 계산.
- PE는 OOS 검증됨: x0.8을 x1.0보다 −0.00037 예측 = 실측 −0.00038(1e-5). 도구 `sleep_proxy/goal056_lastshot.py`.

## 결정적 결과: PE-Δ ↔ worst-loose 경계가 존재
- **tier1(PE Δ≤−0.0015 & worst≤+0.0015) = 0개. tier2(PE Δ≤−0.0010 & worst≤+0.0015) = 0개.**
- 전 grid에서 worst-loose ≤ +0.0015 인 건 **단 2개: C1(PE −0.00048), C2(PE −0.00082)**.
- PE Δ를 −0.0010 아래로 내리는 모든 config는 worst-loose +0.0024~+0.0049(캡 2~3배 초과). 즉
  **공격성을 더 키워도 worst-case 위험만 급증, 예측 public은 ~0.5602가 천장**.
- 사용자 제안 corners 전부 캡 초과: corner1 worst+0.00239 / corner3 +0.00322 / corner4 +0.00225.

## 후보 비교 (predLB = 0.5615 + PE Δ)
| 후보 | (Q2i,Q2f,Q3i,Q3f) | PE Δ | worst-loose | fcv7 | predLB |
|---|---|---|---|---|---|
| anchor(측정) | uniform 0.8 | 0 | −0.00171 | 0 | **0.56153(실측)** |
| C1 | (0.6,1.0,0.6,1.0) | −0.00048 | **−0.00048(safe)** | −0.0016 | ~0.5611 |
| **C2 = 채택** | (0.4,1.2,0.4,1.2) | −0.00082 | +0.00089 | −0.0024 | ~0.5607 |
| grid 최강(캡초과) | (0.2,1.6,0.6,1.6) | −0.00135 | +0.00368 | −0.0004 | ~0.5602 |

## 선택: FINAL = C2
- 사용자 규칙대로 tier1/tier2 부재 → C1 vs C2 비교 → C2(더 큰 PE Δ·fcv, worst-loose 캡 내).
- **정직 경계**: C2 천장 ~0.5607, grid 절대천장 ~0.5602. **어느 것도 수상권(~0.54) 도달 못함**(two-bucket 최대 이득 ~0.0013 « 필요 0.02). 35등 탈출 불가가 정직한 평가. 그래도 "캡 내 가장 공격적" Pareto점 = C2.
- 최종 파일 = `submission_twobucket_int0p4_fut1p2_50deaad1_uploadsafe.csv` (업로드안전 재검증 OK, Q2/Q3만 변경, Q1/S1-4 anchor 동일). 중복본 삭제, 최종 CSV 1개 유지.

## 제출 결정
```
FINAL_LAST_SHOT = submission_twobucket_int0p4_fut1p2_50deaad1_uploadsafe.csv
REASON = 캡(worst-loose≤+0.0015) 내 Pareto-최적 공격 후보; 전 grid 전수탐색이 더 나은 후보 부재를 증명
EXPECTED_LB ≈ 0.5607 (PE OOS-calibrated; C1 ~0.5611 / anchor 측정 0.5615 와의 trade)
RISK = worst-loose +0.00089(양수, anchor 미지배). last=final 방식이면 worst-case ~0.0026 private 손해 가능.
       플랫폼이 최종본 선택 허용시 anchor 보존 가능→하방≈0. 단 성공해도 수상권 미달.
```

## 🚨 실측 결과 (2026-06-18): C2 제출 → **public LB 0.5633508973 = DEAD**
- anchor 0.5615333471 대비 **+0.0018175502 악화**(예측 ~0.5607과 +0.00264 정반대). **two-bucket 레버 전 방향 폐기.**
- **결정적 학습 = 0614c global-prior 가설 실측 확정**: Q2/Q3 overshoot의 LB 이득은 interleaved에도 필요한
  **글로벌 레벨 효과**. C2가 interleaved σ를 0.8→0.4로 줄여 62.4% 다수에서 up-shift 제거 → LB 악화.
  within-subject future/interleaved 차등은 **test 라벨에 전이 안 됨**(forward-CV/PE가 본 신호=비전이).
- **PE 방법론 한계 노출**: x1.0→x0.8 단일 OOS점은 *동일 uniform-σ 축*이라 보간 정확했으나, two-bucket=
  *per-subject 재분배 축*은 PE가 외삽 오판(per-subject AVG만 봄). **단일 근접 OOS 검증 ≠ 새 축 신뢰성.**
- **확정: uniform overshoot×0.8 (=0.5615) 가 진짜 최적·정보바닥.** 30+접근 + two-bucket 전수탐색 모두 격파 실패.
- 폐기: C1/C2 CSV 삭제. anchor `submission_overshoot_x0p8_330ef1a1_uploadsafe.csv` = 최종 best 보존.
  스크립트(sp_05/06/07, goal056)·문서는 방법론 기록으로 유지.
- **⚠️ 행동 필요**: 플랫폼 최종 제출본 선택이 anchor 0.5615를 가리키는지 확인(C2 0.5633 아님).
