# GOAL_059 — Prediction-bank ensemble reset

작성일 2026-06-18 · 마지막 성능 캠페인. 새 모델 X. 기존 OOF/submission 전부 prediction bank로 모아
anchor(0.5615333471, overshoot×0.8)와 다른 방향만 골라 target-wise logit 앙상블 3종(conservative/
balanced/aggressive) 생성. CSV는 anchor와 충분히 다르고 + CV 안 망가지고 + 실패방향 비중복 + 전이게이트
PASS_LEVEL일 때만. 도구 `sleep_proxy/goal059_prediction_bank.py`.

## bank
- 115 OOF 멤버: zoo 108(cat/lgbm/xgb × full/data_enhanced/cross_modality × 분할), knn 2, rocket 1, meta 2, failed 2(seqcnn/mislstm).
- 256 test submission CSV + ledger 29(공개 LB 알려진 것). **ledger 전부 anchor보다 나쁨**(best 0.5647 > 0.5615). bank에 anchor보다 LB 좋은 점 없음.
- anchor는 honest CV서 base보다 **나쁨**(+0.00143) — LB 우위는 CV 불가시 global overshoot.

## 3 앙상블 결과 (anchor backbone + 탈상관·비실패 residual)
| style | total_w | CV mean Δ vs anchor | L2 dist | 전이게이트 | PASS_LEVEL | CSV |
|---|---|---|---|---|---|---|
| conservative | 0.10 | **−0.00298** | 0.418 | Q2/Q3 LOTTERY, S2 DATE_BOUND, S4 LOTTERY, Q1/S1/S3 submargin | **NONE** | ✗(거리부족) |
| balanced | 0.25 | **−0.00702** | 1.039 | 동일 | **NONE** | ✗(전이X) |
| aggressive | 0.50 | **−0.01263** | 2.053 | 동일 | **NONE** | ✗(전이X) |

## 결정적 발견 = CV-LB 괴리 교과서 사례
- 3종 모두 **OOF CV는 크게 개선**(aggressive 0.600→0.587, −0.013 = 전이된다면 거대). 하지만 전이게이트는
  전부 **LOTTERY / DATE_BOUND / sub-margin** — **PASS_LEVEL 전이 타깃 0개.**
- 즉 bank 앙상블의 CV 이득은 **전이 안 되는 종류**(per-row lottery + date-bound). two-bucket이 CV 좋고 LB
  실패한 것과 동일 패턴. aggressive를 제출하면 0.587이 아니라 ~0.57+(anchor보다 나쁨)일 것 = 게이트가 정확히 차단.
- [[experiment_log_0614]] "108-OOF zoo 일괄 vet = 전부 lottery" 재확정. bank에 새 전이 레버 없음.

## 결론
```
GOAL_059_RESULT = FAIL (CV 개선은 전부 비전이 LOTTERY/DATE_BOUND; PASS_LEVEL 0)
BEST_CANDIDATE = none → anchor submission_overshoot_x0p8_330ef1a1 (0.5615333471) 유지
CSV_WRITTEN = 0 (게이트 정상 차단)
REASON = prediction bank의 어떤 다양성 앙상블도 OOF CV만 개선할 뿐 전이게이트서 LOTTERY/DATE_BOUND.
         ledger상 bank 전 멤버가 anchor보다 LB 나쁨. 전이되는 유일 레버(global overshoot)는 anchor가 이미 보유.
```
**최종**: Q-side(overshoot/two-bucket)·S-side(sleep recon)·ensemble-side(bank reset) 3축 모두 소진.
정직한 바닥 ≈0.56, anchor 0.5615333471 = 최종 best. 남은 산출물=검증 방법론(CV-LB 괴리·정보천장·placebo·전이게이트).
