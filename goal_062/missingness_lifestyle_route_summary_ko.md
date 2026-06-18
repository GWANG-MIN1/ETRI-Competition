# 결측/생활리듬 기반 Base-Error Route 요약

## 전제

- `MEMORY.md`는 프로젝트 전체에서 발견되지 않았다.
- 대신 `experiment_log.md`, `failed_hypotheses.md`, `handoff6.md`를 확인했다.
- 기존 기록의 핵심 힌트는 “raw sensor 값보다 observation process/missingness가 residual과 약하게 상관 있다”였다.
- 그래서 이번에는 수면을 직접 맞히는 대신, **base가 누가/언제 크게 틀리는지**를 결측과 생활 리듬으로 찾는 실험을 수행했다.

## 실행한 검증

1. `missingness_rhythm_error_probe.py`
   - feature store에서 원본 lifelog 파생 결측/리듬 feature `4,951`개를 추려 사용.
   - subject deviation/calendar를 추가하고 최종 `320`개 feature로 축소.
   - signed residual 모델, high-error detector, risk shrink를 OOF로 검증.
   - 전이 게이트와 placebo shuffle까지 확인.

2. `future_lifestyle_rule_probe.py`
   - 더 엄격한 forward-only 검증.
   - fit: 앞쪽 날짜 `271`행.
   - tune: 중간 날짜 `79`행.
   - future: 마지막 날짜 `100`행.
   - 앞쪽 날짜로 방향/강도/규칙을 고르고 future에서만 평가.

3. `subject_prior_phase_audit.py`
   - per-subject residual 보정이 tune/future/rolling next-block에서 같은 방향인지 감사.
   - phase sign reversal 여부 확인.

## 발견 1: 결측/리듬은 “오차 위치”를 약하게 잡는다

OOF 기준으로는 `signed subject-level` 보정이 평균 `-0.003604`까지 내려갔다.

하지만 high-error detector 품질은 약했다.

```text
interleaved mean high-error AUC     0.5117
subject-holdout mean high-error AUC 0.4948
future-tail mean high-error AUC     0.4803
```

즉, 같은 subject/date 구조 안에서는 약간 보이지만 subject-holdout/future에서는 거의 무너진다.

## 발견 2: 가장 강한 생활리듬 feature는 야간 앱 사용량이다

강한 correlation 상위권은 대부분 다음 계열이었다.

- `mUsageStats sleep_00_09 app_total_time`
- `mUsageStats sleep_03_06 app_total_time`
- `mUsageStats sleep_06_09 app_total_time`
- `q_evening app_mean_time`

해석:

- raw accelerometer가 없어도, 야간/새벽 스마트폰 사용 패턴은 base residual과 약한 관계가 있다.
- 하지만 이 관계는 target/date phase에 따라 방향이 뒤집혀 직접 규칙으로 쓰면 위험하다.

## 발견 3: forward-only에서 feature rule은 실패했다

앞쪽 날짜에서 선택한 threshold lifestyle rule은 tune에서는 크게 좋아 보였지만 future에서 악화했다.

```text
threshold_lifestyle_rule_tuned future mean_delta = +0.021793
feature_signed_ridge_tuned     future mean_delta = +0.056025
feature_risk_shrink_tuned      future mean_delta = -0.000526
```

결론:

- “언제 base가 틀리는지”를 야간 앱 사용량으로 직접 규칙화하는 것은 현재 검증에서는 overfit이다.
- risk shrink만 아주 작게 살아 있지만 크기가 너무 작다.

## 발견 4: 유일한 glimmer는 “누가 틀리는지”, 즉 subject-level residual reversal

forward-only future에서 가장 나은 것은 feature가 아니라 subject별 과거 residual의 반대 방향 보정이었다.

```text
subject_prior_residual amp=-0.4
future mean_delta = -0.003798
Q3 delta          = -0.027119
```

하지만 phase audit에서 안정성이 없었다.

```text
fit -> tune best amp              +0.2, mean_delta -0.000676
fit+tune -> future best amp       -0.4, mean_delta -0.003798
rolling previous blocks -> next   0.0,  mean_delta  0.000000
```

즉, future에서는 subject residual이 뒤집히는 듯하지만, tune과 rolling block에서는 그 방향을 고를 근거가 없다.

## 상한 계산

future label을 몰래 보고 target별 최적 amp를 고르는 oracle 상한도 계산했다.

```text
Q1 best -0.008702
Q2 best -0.004808
Q3 best -0.046370
S1 best -0.008205
S2 best -0.001457
S3 best  0.000000
S4 best -0.002484
mean    -0.010289
```

현재 public anchor `0.561533`에서 `0.54`까지 필요한 개선은 약 `-0.0215`다.

따라서 이 subject-residual 계열은 future label을 후견지명으로 써도 상한이 `-0.0103` 정도라서, 단독으로는 0.54에 닿지 못한다.

## 결론

정직하게 말하면:

- 결측/생활리듬은 base-error 위치를 **조금** 설명한다.
- 가장 살아 있는 신호는 “언제”보다 **누가**, 특히 subject별 residual drift/reversal이다.
- 그러나 그 reversal 방향은 train 내부에서 안정적으로 선택되지 않는다.
- robust 0.54 경로로는 아직 부족하다.

그래도 5% 확률의 도전 경로를 꼽으면 다음 하나다.

## 5% 도전 경로

**Q-side subject residual reversal micro-probe**

- train에서 subject별 base residual을 계산한다.
- public test에서는 그 residual의 반대 방향으로 Q1/Q2/Q3를 아주 작게 민다.
- S1/S3는 건드리지 않는다.
- S4는 매우 작게만 포함하거나 제외한다.

이유:

- train future-tail에서 Q3가 가장 크게 반응했다.
- `amp=-0.4`는 future에서 평균 `-0.0038`, Q3에서 `-0.0271`을 보였다.
- 그러나 tune/rolling에서는 선택 근거가 없으므로 final replacement가 아니라 public 정보 탐사용 probe로만 써야 한다.

위험:

- tune과 future의 sign이 반대다.
- rolling block 검증은 0 이동이 최선이다.
- 성공해도 기대 개선은 `0.003~0.006` 수준이고, 0.54까지의 전체 gap을 닫지는 못한다.

## 저장된 산출물

- `outputs/missingness_rhythm_error_probe_report.md`
- `outputs/missingness_rhythm_error_probe_grid.csv`
- `outputs/missingness_rhythm_error_probe_gate.csv`
- `outputs/missingness_rhythm_error_probe_detection.csv`
- `outputs/missingness_rhythm_error_probe_future.csv`
- `outputs/missingness_rhythm_error_probe_feature_corr.csv`
- `outputs/future_lifestyle_rule_probe_report.md`
- `outputs/future_lifestyle_rule_probe_grid.csv`
- `outputs/future_lifestyle_rule_probe_details.csv`
- `outputs/subject_prior_phase_audit.md`
- `outputs/subject_prior_phase_audit.csv`
- `outputs/subject_prior_future_oracle_bound.csv`
