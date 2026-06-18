# GOAL_060 — External-data S transfer (Walch sleep-accel) : FAIL (sensor information ceiling)

작성일 2026-06-19 · 목적: MESA 심사 대기 회피, 즉시 가능한 외부 PSG-라벨 데이터로 S 천장이
"feature 정보 부재"인지 "약한 감독/데이터 부족"인지 결판. = 0.54 향한 마지막 외부 레버.

## 데이터
- **Walch `sleep-accel`**(PhysioNet, ODC-By 1.0, Open): Apple Watch 손목 **accel + HR(BPM) + steps + PSG 수면라벨**, 31명.
- 다운로드: heart_rate/ + steps/ + labels/ 만(motion/ 2GB 제외), 93파일 ~4.7MB. `data/walch_sleep_accel/`.
- 라벨=30초 epoch(0=wake/1=N1/2=N2/3=N3/5=REM). HR=초단위 불규칙 BPM. steps=구간 step수.
- 전이 채널(ETRI 가용): **HR-bpm 통계 + step활동**, 60초 epoch(ETRI 분 단위와 정합). raw accel은 ETRI 부재로 제외.

## Stage 1 — Walch 자체 sleep/wake sanity (LOSO 31fold)
- HR+steps@60s sleep/wake AUC: **LR 0.639 / GBM 0.586** = WEAK(<0.75).
- feature importance: HR가 전부(hr_rel 841/hr_mean 700/hr_std 528/...), **step_sum 55, step_active 0** = **step 채널 무신호**(밤엔 걸음=0; actigraphy 신호인 raw-accel 미세움직임이 없음).
- → ETRI-가용 채널만으론 sleep/wake조차 0.64가 한계.

## Stage 2 — ETRI 전이 (Walch 학습 모델 → ETRI 밤 proxy → S, 346/450 커버)
| S | proxy | AUC | 게이트 |
|---|---|---|---|
| S1 | TST | **0.537** | NO_GAIN (d_full +0.0056) |
| S2 | SE | **0.500** | NO_GAIN (+0.0086) |
| S3 | SOL | **0.513** | NO_GAIN (+0.0059) |
| S4 | WASO | **0.534** | PASS_LEVEL~submargin (+0.0090, 미인증) |
- **PASS_LEVEL = NONE. d_full 전부 양수(블렌드 시 악화).** 가장 robust한 aggregate인 S1=TST도 0.537(≈chance).

## 결론
```
GOAL_060_RESULT = FAIL (sensor information ceiling, not supervision/data-size)
BEST = anchor submission_overshoot_x0p8_330ef1a1 (0.5615333471) 유지
REASON = 외부 PSG 정답 + HR-bpm modality 정확 일치로 학습·전이해도 ETRI S proxy AUC ≈ chance(0.50-0.54).
         천장 원인 = raw 가속도(미세움직임) 부재. ETRI 모션=분단위 step수(밤엔 0)라 sleep/wake 신호 없음.
         HR만으론 sleep/wake 0.64 한계 → 파생 S메트릭 ≈ chance. S-side 정보천장 확정.
```
**진단 가치(논문)**: GOAL_058 천장이 데이터부족이 아니라 **센서 modality 한계**임을 외부 gold-standard로 입증.
ETRI가 0.54로 가려면 raw accelerometry가 필요(현 데이터엔 없음). 도구: `outputs/walch_stage1_sanity.py`, `walch_stage2_transfer.py`.

## 전체 종합 (4축 소진)
Q(overshoot 포화·two-bucket LB실패) · S(센서 정보천장, GOAL_058+060 확정) · ensemble(CV-LB 괴리, GOAL_059) ·
external-data(GOAL_060, 센서 천장 재확인) → **0.54 경로 없음, 정직 바닥 ≈0.56, 최종 best = anchor 0.5615333471.**
