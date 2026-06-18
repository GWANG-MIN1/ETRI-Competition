# HS-JEPA / DACON #236690 — Handoff 6 (2026-06-18 세션 종합)

> 대회: 제5회 ETRI 휴먼이해 AI **논문**경진대회 (수면/스트레스) · DACON #236690 · 마감 **2026-06-26**, 3제출/일
> 직전: handoff5.md(GOAL_055 "진짜 JEPA 월드모델"). 이 세션: 전략doc 실행 → **3개 성능 캠페인(GOAL_056/058/059)
> 전부 정직검증 후 FAIL** → 0.54 경로 없음 + 정보바닥 ≈0.56 최종 확정. **새 제출 CSV 0개(2개 만들었다 폐기·삭제).**
>
> **한 줄 요약: Q-side·S-side·ensemble-side 3축 모두 소진. 최종 best = anchor 0.5615333471(보존). 0.54는
> 통계+정보이론상 도달 불가. 남은 실질 산출물 = 검증 방법론(논문).**

---

## 0. 대회 / 메트릭 (불변)
- 250 test rows × 7 binary targets **평균 logloss(낮을수록 좋음)**. public LB ≈ 50% 서브셋.
- 7타깃: **Q1/Q2/Q3** = 주관 설문(피로/스트레스/수면질; 개인평균 대비; day-to-day self-corr≈0 = per-row 노이즈, 느린 drift만 전이).
  **S1=TST / S2=SE / S3=SOL / S4=WASO** = Withings **침대센서** 객관 NSF 임계(미제공; stable trait).
- 구조: test=train과 동일 10명. **62.4% interleaved / 37.6% future**. train 450 / test 250 rows. n=10이 통계 한계의 핵심.

## 1. 목표 & 정직한 도달성
- **목표: public LB 0.54~0.55** (사용자 희망, 수상 안정권). 현재 ~15~35등권, 상위 5등 ≈ 0.54(누수 의심·재현 불가).
- **정직한 평가: 0.54 도달 불가.** 정보바닥 ≈ **0.56**. 0.54는 그 아래 0.02 = 2번째 jackpot급 전이신호 필요한데
  Q·S·ensemble 전 축에서 없음(아래 §3·§5). per-subject 완벽보정 floor 0.5904, per-subject calib floor 0.6182.

## 2. 현재 best / 제출 실측
| 항목 | 값 |
|---|---|
| **public LB best (제출됨, 최종)** | **0.5615333471** = `submission_overshoot_x0p8_330ef1a1_uploadsafe.csv` |
| 구성 | FS base + **0.8 × per-subject Q2/Q3 recency logit-step**(uniform). S/Q1 무변. |
| anchor x1.0 (이전 제출) | 0.5619100863 |
| FS base (FrontierSilence) | 0.56773 |
| 이번 세션 실패 제출 | two-bucket C2 → **0.5633508973**(anchor보다 +0.0018 악화) |

## 3. 이번 세션 진행 실험 (전부 FAIL → 폐기)
### GOAL_056 — Q2/Q3 two-bucket overshoot (`sleep_proxy/sp_05~07`, `goal056_lastshot.py`, `GOAL_056_*.md`)
- 착상: anchor는 전 행에 uniform σ=0.8. interleaved(드리프트↓→σ작게)/future(드리프트↑→σ크게)로 차등(bucket=관측값=전이클래스).
- forward-CV + LB-보정 PE(OOS 1e-5 검증) **둘 다 통과**, target-specific 4-sigma 625 전수탐색까지 수행.
- **실측 제출(C2) → 0.5633509 = FAIL.** 0614c **global-prior 가설 실측 확정**: overshoot 이득은 interleaved에도 필요한
  글로벌 효과 → interleaved σ 줄이면 악화. forward-CV/PE 둘 다 −예측, 실측 +. **CV-LB 괴리 재현.** C1/C2 CSV 삭제.

### GOAL_058 — Sleep-JEPA hidden sleep architecture decoder for S2/S3/S4 (`sleep_proxy/goal058_s3_jepa_probe.py`, `GOAL_058_*.md`)
- 착상: HS-JEPA latent로 수면 이벤트 복원 → TST/SE/SOL/WASO proxy → S 직접 겨냥(MIL).
- **정보 천장으로 FAIL(방법 아니라 센서 물리).** 4경로 독립 확정: ① deterministic ceiling(복원메트릭 AUC S2 .509/S3 .523/
  S4 .523 ≈ 동전, GLOBAL ceiling=base_rate, S3 커버 55%) ② actigraphy 검증식(Sadeh/Cole-Kripke gold-standard,
  S3 AUC 0.415, transferable_mean_gain +0.00000) ③ seqcnn(야간 시퀀스 지도학습 전타깃 DEAD) ④ frozen-JEPA→S3 probe
  (real_z가 time-shuffle/rowperm placebo보다 **나쁨**, +0.0099 악화, gate NO_GAIN). **full MIL decoder는 빌드 안 함**(probe FAIL).

### GOAL_059 — prediction-bank ensemble reset (`sleep_proxy/goal059_prediction_bank.py`, `GOAL_059_*.md`)
- 115 OOF(zoo 108 + knn/rocket/meta + seqcnn/mislstm) 모아 anchor와 탈상관·비실패 방향만 → logit 앙상블 3종.
- 3종 모두 **OOF CV 크게 개선**(aggressive 0.600→0.587, −0.013) **그러나 전이게이트 전부 LOTTERY/DATE_BOUND,
  PASS_LEVEL 0** = CV-LB 괴리 교과서. ledger 29개(공개LB 알려진 submission) **전부 anchor보다 나쁨**. **CSV 0개(게이트 차단).**

### (세션 초) 전략doc A/B/C 검증 (`sleep_proxy/sp_00~04`, `STRATEGY_DOC_VERIFICATION.md`)
- A=Q2/Q3 JEPA게이트 / B=S2/S4 sleep-proxy(17 night proxy) / C=캘리브 → **전부 DEAD**(base가 흡수/placebo 미격파/best 중복).

## 4. 폐기한 실험 graveyard (누적 — 재시도 금지)
- **Q recency/overshoot 튜닝**: τ/σ 스윕, two-bucket, target-specific 4-sigma. 전이되는 건 uniform overshoot×0.8뿐(=anchor).
- **S-target reconstruction**: 물리복원, actigraphy(Sadeh/Cole-Kripke), latent-sleep IRT, NSF 임계 역설계, S-shrink/de-shrink,
  S4 routine-anomaly glimmer, S2-down drift, sleep-window proxy 추가 — 전부 정보천장/subsumed/date-bound.
- **표현학습**: JEPA z[96] concat(placebo가 이김), world-model JEPA 증폭 6레버, within-subject-z, seqcnn 1D-CNN, rocket, KNN — 전부 비전이.
- **앙상블/스태킹**: 108-OOF zoo vet(전부 lottery), multi-target stacker, deep blend(mislstm 0.5663), decorrelated blend, bank reset(GOAL_059).
- **캘리브레이션**: temperature scaling, clipping(이미 범위 안), subject-shrink(Q2만 best와 중복), per-target. 전부 무효/중복.
- **public-LB 끼워맞추기/polytope tomography**: candidate selection 미해결, frontier 0.577 plateau, selector 해상도 부족. 금지.

## 5. 한계 — 왜 0.54가 불가능한가 (정직)
1. **통계**: n=10 피험자. Q는 day-to-day self-corr≈0(per-row 비전이), 전이되는 건 느린 level/drift뿐인데 base가 이미 흡수.
2. **정보이론(S)**: S2/S3/S4 = 침대센서 메트릭의 NSF 임계인데 손목워치+폰 5분bin은 그 메트릭을 못 봄(복원 AUC ≈ 동전). decoder는 입력에 없는 정보 생성 불가.
3. **CV-LB 괴리**: per-row/level CV 이득은 LB 전이 안 됨. 유일 LB-무버 = global prior shift, robust 최적 = overshoot×0.8(이미 anchor). 새 인증 레버 0.
4. **gate-fooling placebo 법칙**: 노이즈가 물리신호보다 강한 가짜 PASS 빈발 → placebo·전이게이트 필수.

## 6. 핵심 검증 방법론 (논문 산출물 — 실질 가치)
- **전이게이트** `outputs/gate_transfer_vet.py`: level/resid 분해 + interleaved-CV + future-block + placebo → PASS_LEVEL/LOTTERY/DATE_BOUND/NO_GAIN 판정.
- **PE 폴리토프** `polytope_eval.py`: 실측 public-LB ledger로 제약된 posterior. OOS 1e-5 검증(x0.8를 x1.0서 예측). 단 새 축(two-bucket)엔 오판 → 한계도 기록.
- **forward-CV**(미래행) + **5종 placebo**(time/subject/z-shuffle/reverse/random) + **deterministic ceiling**(정보 상한).
- 결정화된 발견: CV-LB 괴리 인과, S 정보천장, placebo 법칙, global-prior가 유일 전이축.

## 7. 다음 세션 권장 / 금지
- **권장**: ① 최종 제출 선택본을 **anchor 0.5615**로 확정(플랫폼이 last=final이면 C2 0.5633 아님 반드시 확인). ② 논문(검증 방법론·negative result) 정리.
- **금지(전부 실측·정보이론으로 사망)**: Q overshoot/two-bucket 재시도, S-target sleep reconstruction, JEPA z concat, sleep-window 피처 추가,
  deep blend/bank 앙상블 재시도, calibration 재탐색, public-LB tomography, "더 큰 encoder". CV만 보고 제출 금지(반드시 전이게이트+PE).

## 8. 파일 인벤토리 / 메모리 포인터
- 트래커: `GOAL_056_LASTSHOT.md`, `GOAL_058_SLEEP_JEPA_MIL_DECODER.md`, `GOAL_059_PREDICTION_BANK_ENSEMBLE_RESET.md`, `sleep_proxy/STRATEGY_DOC_VERIFICATION.md`.
- 스크립트: `sleep_proxy/sp_05~07`(two-bucket+audit), `goal056_lastshot.py`, `goal058_s3_jepa_probe.py`, `goal059_prediction_bank.py`;
  증거 `outputs/goal054_{label_def,actigraphy_sleep}.py`; 도구 `outputs/{goal054_kit,gate_transfer_vet}.py`, `polytope_eval.py`.
- best 보존: `submission_overshoot_x0p8_330ef1a1_uploadsafe.csv`(0.5615333471). 빌드: `save_overshoot_x08.py`.
- 메모리: `experiment_log_0618b_twobucket`(GOAL_056) / `0618c_goal058`(S-jepa) / `0618d_goal059`(bank) / `0618_strategydoc`(A/B/C) /
  `0614c`(정보바닥·CV-LB) / `0615`(완벽보정 floor) / `0616_worldmodel_jepa`(JEPA) / `dacon_submission_checklist`.
