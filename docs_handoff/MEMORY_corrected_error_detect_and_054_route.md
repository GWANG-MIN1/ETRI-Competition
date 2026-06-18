# MEMORY 반영 정정 보고서

## 정정

처음에는 `Documents/Codex` 아래만 검색해서 `MEMORY.md`를 못 찾았다.
실제 파일은 아래 경로에 있었다.

```text
C:\Users\박광민\.claude\projects\C--Users----\memory\MEMORY.md
```

이 파일은 인덱스였고, 핵심 내용은 링크된 실험일지에 있었다.

## 읽은 핵심 메모리

- `experiment_log_0619b_error_detect.md`
- `data_modality_external_transfer.md`
- `experiment_log_0614c.md`
- `MISLSTM_README.md`
- `NOTION_mislstm_writeup.md`
- `GOAL_062_ERROR_DETECT.md`

## 결측 + 생활리듬 error-detect의 올바른 결론

내가 방금 새로 만든 `missingness_rhythm_error_probe.py`보다, 기존 `GOAL_062`가 같은 질문을 더 정확히 검증했다.

`GOAL_062`의 결론:

```text
Phase1 = PASS
risk -> base row_loss Spearman:
held-out +0.164
future   +0.187
placebo  +0.002
```

즉, **base가 confident-wrong인 행은 결측 + 생활리듬으로 예측 가능하다.**

하지만:

```text
Phase2-3 = FAIL
risk-gated shrink가 random-gated shrink를 못 이김
transfer-gate PASS_LEVEL NONE
CSV 0
```

즉, **틀릴 위험은 찾지만, 어느 방향으로 고쳐야 하는지는 못 찾는다.**

최종 판정:

```text
detectable but not correctable
```

## 내가 만든 subject-reversal probe에 대한 정정

이번 작업에서 생성한 아래 파일들은 최종 후보가 아니다.

- `outputs/submission_probe_subject_reversal_q3_m0p2.csv`
- `outputs/submission_probe_subject_reversal_q123_m0p2.csv`
- `outputs/submission_probe_subject_reversal_q123s4_m0p4.csv`

이들은 `MEMORY` 기준으로 보면 **검증된 제출 후보가 아니라 위험한 public probe**다.

이유:

- `GOAL_062`가 이미 risk-gate 교정 실패를 확인했다.
- subject residual reversal은 tune/future sign이 뒤집힌다.
- rolling next-block에서는 0 이동이 최선이다.
- 교정 방향이 안정적으로 선택되지 않는다.

따라서 이 파일들은 **anchor 대체용으로 사용하면 안 된다.**

## 0.54 경로에 대한 정정

`MEMORY`를 읽기 전 내 결론은 “정보가 없어서 거의 0.56 바닥” 쪽에 너무 치우쳤다.

정확한 역사:

1. 0.54는 레벨/캘리브레이션으로는 불가능하다.
2. 0.54가 가능하려면 새로운 per-row 모델 클래스가 필요하다.
3. `MEMORY`에서 그 후보로 발견한 것이 MIS-LSTM 계열이다.
4. MIS-LSTM은 실제로 Q2/Q3에서 직교 per-row 신호를 보였다.
5. 그러나 제출 실측에서는 `submission_mislstm_conservative`가 public `0.5663`으로, 현재 anchor `0.561533`을 이기지 못했다.

따라서 현재 결론은 다음이다.

```text
0.54로 가는 유일하게 말이 되는 방향 = 더 faithful한 MIS-LSTM/딥 시퀀스 재현
현재까지 검증된 안전 최종 = overshoot x0.8 anchor 0.561533
결측/생활리듬 error-detect = 논문가치 있는 진단 신호, 제출 후보 아님
```

## 지금 남은 5% 도전 경로

결측 규칙이나 subject-reversal probe가 아니다.

남은 5%는 아래다.

```text
MIS-LSTM faithful 재현 재시도
SEResNeXt101_32x4d + CBAM + 2-layer LSTM + subject embedding + UALRE ensemble
평가 기준은 Macro-F1이 아니라 logloss calibrated OOF + transfer gate
```

이미 존재하는 실행 자산:

- `C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs\goal054_mislstm_colab.py`
- `C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs\goal054_mislstm_tensors.npz`
- `C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\outputs\goal054_mislstm_eval.py`
- `C:\Users\박광민\Documents\Codex\etri_team\final_hsjepa_candidates\MISLSTM_README.md`

실행 원칙:

1. Colab/GPU에서 `goal054_mislstm_colab.py`를 돌린다.
2. `goal054_mislstm_oof_ensemble.csv`, `goal054_mislstm_test_ensemble.csv`를 가져온다.
3. 로컬에서 `goal054_mislstm_eval.py`로 logloss, nested blend, transfer gate를 본다.
4. `PASS_LEVEL`이 없으면 제출하지 않는다.
5. `PASS_LEVEL`이 Q2/Q3에서 floor-clearing으로 뜰 때만 FS/anchor blend 후보를 만든다.

## 최종 판단

현재 anchor `submission_overshoot_x0p8_330ef1a1_uploadsafe.csv`는 유지한다.

결측 + 생활리듬은:

- base 오류 탐지 신호로는 성공
- 보정 방향/제출 후보로는 실패

0.54 도전은:

- 결측 rule이 아니라 MIS-LSTM faithful 재현에서만 5% 확률이 있다
- 단 기존 MIS-LSTM 제출 `0.5663` 실패 때문에 기대값은 낮다
- 그래도 “포기하지 않고 더 파볼” 유일한 기술적 축은 이쪽이다
