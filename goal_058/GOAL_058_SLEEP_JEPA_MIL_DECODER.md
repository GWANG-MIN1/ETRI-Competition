# GOAL_058 — Sleep-JEPA hidden-sleep-architecture MIL decoder for S2/S3/S4

작성일 2026-06-18 · 전환 동기: GOAL_056(Q2/Q3 two-bucket) 실측 실패(C2 0.5633509). Q로는 0.54 불가 →
S2/S3/S4 구조적 gain 필요. 가설: HS-JEPA latent로 hidden sleep architecture(수면 이벤트) 복원 →
TST/SE/SOL/WASO proxy → S 타깃 직접 겨냥. 금지: Q overshoot/two-bucket 재시도, JEPA z 단순concat,
sleep-window 평균피처만 추가, 전타깃 동시변경, 누수/LB끼워맞추기.

## ⚠️ FEASIBILITY GATE 먼저 — 이 thesis는 이미 다각도로 구현·검증됨
GOAL_058의 핵심(야간센서 → 수면/각성 hypnogram → TST/SE/SOL/WASO → S 타깃 → placebo vet)은 이미 구현돼 있음:
- `outputs/goal054_physical_S.py` — 야간센서서 S메트릭 물리복원(deterministic predictor).
- `outputs/goal054_actigraphy_sleep.py` — **PSG-검증 actigraphy(Sadeh 1994·Cole-Kripke 1992)** hypnogram →
  TST/SE/SOL/WASO → NSF 임계 calibrate → blend/override → gate + noise placebo. = GOAL_058의 rule-based 버전.
- `outputs/goal054_latent_sleep_generative.py` — 단일 잠재 수면인자 Z → 4 S를 item-임계로 생성(IRT). 4 S 공통원인 결합.
- `outputs/goal054_label_def.py` — NSF 임계 역설계 + **deterministic CEILING**(센서가 bed-sensor 메트릭을 복원 가능한가).

## 🚨 결정적 증거 1 — DETERMINISTIC CEILING (정보 천장, goal054_label_def.py 실측)
"우리 야간센서로 복원한 수면메트릭이 진짜 S 라벨을 얼마나 맞히나 (AUC)" — 이건 **방법이 아니라 입력 정보의 한계**.
어떤 decoder(rule/학습/MIL/JEPA)도 이 천장을 못 넘음(decoder는 입력에 없는 정보를 만들 수 없음).

| 타깃 | 복원메트릭 | 커버리지 | base_rate | GLOBAL ceil | **AUC** | honest CV(전이) |
|---|---|---|---|---|---|---|
| S1=TST | tst_est | 89% | 0.689 | 0.699 | 0.603 | +0.010 |
| S2=SE | se_est | 89% | 0.654 | **0.654(=base_rate)** | **0.509≈chance** | **−0.050(악화)** |
| S3=SOL | sol_est | **55%** | 0.718 | **0.718(=base_rate)** | **0.523≈chance** | **−0.012(악화)** |
| S4=WASO | waso_est | 89% | 0.566 | **0.566(=base_rate)** | **0.523≈chance** | **−0.036(악화)** |

**해석**: S2/S3/S4의 복원 수면메트릭 AUC ≈ 0.51~0.52 = **거의 동전던지기**. GLOBAL 임계 ceiling = base_rate
(다수class보다 나을 게 없음). S3는 커버리지 55%(밤 45%는 sleep onset 추정조차 불가). within-subj ceiling(0.75~0.83)은
같은 라벨에 임계 과적합한 OPTIMISTIC 수치(비전이). → **Withings bed-sensor가 보는 것(수면단계별 미세움직임·심박,
실제 wake epoch)을 손목워치+폰 5분bin은 못 봄. S2/S3/S4 hidden sleep architecture는 우리 센서로 복원 불가.**
- 우선순위 1순위였던 S3가 가장 나쁨(커버 55%, AUC 0.523, honestCV 악화).

## 🚨 결정적 증거 2 — 학습 경로도 이미 죽음
- `seqcnn`([[experiment_log_0614b]]): 야간 per-minute **시퀀스 지도학습 1D-CNN** → 전 타깃 DEAD(d_lvl 전부 양수=악화).
- JEPA z[96] 단순concat([[experiment_log_0616_worldmodel_jepa]]): placebo가 z를 이김.
- → 학습된 feature extractor(CNN/JEPA)도 정보 천장 위로 못 올라감. MIL/JEPA decoder의 BEST 출력도 이 천장에 bound.

## 🚨 결정적 증거 3 — actigraphy(PSG-검증식) S vet (goal054_actigraphy_sleep.py 실측)
Sadeh 1994 + Cole-Kripke 1992 (수면연구 gold-standard) hypnogram → TST/SE/SOL/WASO → S, vs base, vs placebo:
- best-proxy AUC: S1 0.420 / S2 0.543 / **S3 0.415** / S4 0.471 (S1/S3/S4는 0.5 미만 = 방향조차 안 맞음).
- blend/override: 전 S 타깃 base보다 악화(S3 blend +0.0018, override +0.0000), conf rows 0.
- 게이트: 전 S NO_GAIN, residGain ≈ placebo (S3 +0.1693 vs placebo +0.1663). **transferable_mean_gain = +0.00000.**
- → 검증된 gold-standard 수면채점도 S 전이신호 0. recon(증거1)·검증식(증거3) 둘 다 동일 벽.

## 🚨 결정적 증거 4 — frozen HS-JEPA → S3 probe (option-2, goal058_s3_jepa_probe.py 실측)
사용자 승인 경량 1발: frozen HS-JEPA z[96] → S3 head(logistic + MLP), time_shuffle/noise/rowperm placebo 대비.
| rep | head | tf_Δ(sign) | future_Δ | 판정 |
|---|---|---|---|---|
| **real_z** | logit | **+0.00993(0.00)** | **+0.01757** | base보다 악화 |
| time_shuffle | logit | −0.00009 | +0.00185 | **placebo가 real보다 나음** |
| rowperm | logit | +0.00096 | −0.00183 | placebo가 real보다 나음 |
| real_z | mlp | +0.00418(0.00) | +0.00599 | base보다 악화 |
- 게이트(real_z S3 blend): **NO_GAIN**, lvlOrth −0.91(base 레벨과 역상관), residGain +0.0092 < placebo +0.0221.
- → frozen JEPA latent은 S3 신호 없음 + 오히려 악화 + placebo보다도 나쁨. (JEPA z=일주기/drift 인코딩=Q축, S와 무관.)
- 3 pass-기준(beats base / clearly beats placebo / gate PASS_LEVEL) 전부 False.

## 평가기준 대비
- 유지조건 "S3 ≥ −0.02 개선" → 모든 경로(ceiling −0.012악화 / actigraphy +0.0018악화 / JEPA-probe +0.0099악화). 도달 불가.
- "placebo보다 확실히 좋아야" → 모든 경로서 real ≈ 또는 < placebo. 전부 NO_GAIN.

## 결론 (확정)
```
GOAL_058_RESULT = FAIL (information ceiling — 방법 아니라 센서 물리)
BEST_CANDIDATE = none → anchor submission_overshoot_x0p8_330ef1a1 (0.5615333471) 유지
CHANGED_TARGETS = none
EXPECTED_LB_RANGE = 0.5615 (S 변경은 악화 EV)
MAIN_SIGNAL = none (S2/S3/S4 복원 AUC ≈ chance 0.51~0.52; frozen JEPA도 S3 악화)
PLACEBO_MARGIN = 음수 (real_z가 placebo보다 나쁨)
REASON = S2/S3/S4는 Withings bed-sensor(TST/SE/SOL/WASO)의 NSF 임계인데, 손목+폰 야간센서는 그 메트릭을
         복원 못함(AUC≈동전). recon/검증actigraphy/seqcnn/frozen-JEPA 4경로 모두 동일 벽 + placebo 미격파.
         decoder는 입력에 없는 정보 생성 불가 → S-side 0.54 경로 없음. S-target reconstruction 방향 중단.
```
산출물: 스크립트 `sleep_proxy/goal058_s3_jepa_probe.py`(probe), 증거 `outputs/goal054_{label_def,actigraphy_sleep}.py`.
full MIL decoder / hidden-state viz / S2·S4 head는 사용자 지시대로 빌드 안 함(probe FAIL → 불필요).
