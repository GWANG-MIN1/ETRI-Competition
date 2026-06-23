# S Hidden-Generator Council Solver

## 확인된 새 기준점

`sleep_world_conflict_rescue` 제출은 public LB `0.5612880941`로 확인됐다.
따라서 이 실험은 그 파일을 새 S base로 고정하고, 그 위에서 남은 S1/S2/S3/S4
숨은 생성식을 복원한다.

## 핵심 가설

S label은 raw sleep episode를 바로 threshold한 값이 아니다. 현재 가장 설득력 있는
생성식은 다음 3층이다.

1. raw sleep episode / boundary state
2. subject별 threshold 또는 measurement convention
3. trajectory, measurement grammar, raw-semantic state가 같은 방향으로 들리는 row-target만 release

이 solver는 2번을 `subject_threshold_calibrated_phase_solver`에서 가져오고,
3번을 trajectory / measurement / raw-semantic atlas의 council agreement로 구현한다.

## Council Cell Diagnostics

| target | cells | mean_abs_desired | mean_council_agreements | mean_council_balance | mean_state_strength |
| --- | --- | --- | --- | --- | --- |
| S1 | 250 | 2.003989 | 4.044000 | -0.136000 | 0.431117 |
| S2 | 250 | 2.702444 | 4.388000 | 0.616000 | 0.382113 |
| S3 | 250 | 3.397596 | 4.840000 | -0.020000 | 0.396826 |
| S4 | 250 | 1.854894 | 4.788000 | -0.128000 | 0.441880 |

## Candidate Summary

| name | changed_s_cells | targets | raw_score | local_score | expected_lb_raw_score | expected_lb_local_score | mean_council_agreements | mean_council_balance | mean_abs_delta | upload_safe | filename |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_threshold_council_balanced_k180 | 180 | S1,S2,S3,S4 | -0.005573 | -0.003233 | 0.555715 | 0.558055 | 7.450000 | 4.250000 | 0.047653 | True | submission_human_life_s_hidden_generator_council_solver_subject_threshold_council_balanced_k180_43bb42cbf8_uploadsafe.csv |
| subject_threshold_council_wide_k320 | 281 | S1,S2,S3,S4 | -0.005787 | -0.004224 | 0.555502 | 0.557064 | 7.220641 | 3.715302 | 0.046676 | True | submission_human_life_s_hidden_generator_council_solver_subject_threshold_council_wide_k320_d40a5c2e83_uploadsafe.csv |
| s3s4_threshold_trajectory_council_k140 | 140 | S3,S4 | -0.003781 | -0.002339 | 0.557507 | 0.558950 | 7.507143 | 4.028571 | 0.047851 | True | submission_human_life_s_hidden_generator_council_solver_s3s4_threshold_trajectory_council_k140_e90f564aa4_uploadsafe.csv |
| dual_negative_sparse_council_k120 | 120 | S1,S2,S3,S4 | -0.002937 | -0.003994 | 0.558351 | 0.557294 | 7.208333 | 3.833333 | 0.045566 | True | submission_human_life_s_hidden_generator_council_solver_dual_negative_sparse_council_k120_eac30ecbfd_uploadsafe.csv |
| opposite_sign_council_search_k180 | 180 | S1,S2,S3,S4 | -0.007556 | -0.003303 | 0.553732 | 0.557985 | 7.322222 | 3.916667 | 0.045244 | True | submission_human_life_s_hidden_generator_council_solver_opposite_sign_council_search_k180_645ed31662_uploadsafe.csv |
| exact_reverse_of_balanced_k180 | 180 | S1,S2,S3,S4 | 0.005573 | 0.003233 | 0.566861 | 0.564521 | 7.450000 | 4.250000 | 0.049471 | True | submission_human_life_s_hidden_generator_council_solver_exact_reverse_of_balanced_k180_6741a0253a_uploadsafe.csv |
| exact_reverse_of_wide_k320 | 281 | S1,S2,S3,S4 | 0.005787 | 0.004224 | 0.567075 | 0.565512 | 7.220641 | 3.715302 | 0.048393 | True | submission_human_life_s_hidden_generator_council_solver_exact_reverse_of_wide_k320_e7bfff17af_uploadsafe.csv |
| two_regime_assignment_union_k360 | 360 | S1,S2,S3,S4 | -0.013129 | -0.006536 | 0.548159 | 0.554752 | 7.386111 | 4.083333 | 0.046449 | True | submission_human_life_s_hidden_generator_council_solver_two_regime_assignment_union_k360_dc3a5f9d53_uploadsafe.csv |
| two_regime_assignment_top240 | 240 | S1,S2,S3,S4 | -0.011601 | -0.006102 | 0.549687 | 0.555186 | 7.379167 | 4.133333 | 0.046054 | True | submission_human_life_s_hidden_generator_council_solver_two_regime_assignment_top240_2f84806702_uploadsafe.csv |
| two_regime_assignment_pairbalanced_k240 | 240 | S1,S2,S3,S4 | -0.011513 | -0.006261 | 0.549775 | 0.555027 | 7.333333 | 4.112500 | 0.046588 | True | submission_human_life_s_hidden_generator_council_solver_two_regime_assignment_pairbalanced_k240_41d62a2c47_uploadsafe.csv |

## 해석

- 좋아지면: S hidden generation은 `subject threshold`만으로 충분하지 않고, 독립적인
  trajectory/measurement/raw-semantic listener가 같은 방향을 말할 때 안전하게 번역된다.
- 나빠지면: subject-threshold 신호는 맞더라도 council agreement가 public action safety로
  직접 이어지지 않거나, 이미 0.561288 conflict base가 추출 가능한 S 구조를 대부분 가져갔다.
- `opposite_sign_council_search`가 좋아지면: S에는 subject-threshold 부호와 반대인
  두 번째 sign regime도 존재할 수 있다.
- `exact_reverse_of_*`가 좋아지면: main council 방향 자체가 틀렸다는 뜻이다. 현재
  offline audit에서는 exact reverse가 명확히 harmful이므로 main sign은 최소한
  public-memory/local-residual 기준에서 반증되지 않았다.
- `two_regime_assignment_*`가 좋아지면: S label은 하나의 subject-threshold polarity가
  아니라 row-target별 A/B sign regime 배정 문제다. 특히 union 후보는 360개 S cell을
  움직이는 큰 주장이고, raw/local expected LB가 각각 `0.548159 / 0.554752`까지 열린다.

## Public Result

`two_regime_assignment_union_k360`를 제출했고 public LB는 다음과 같았다.

```text
0.5595724276
```

비교:

```text
conflict-rescue base: 0.5612880941
realized gain:       -0.0017156665
```

이 결과는 `two_regime_assignment` 가설을 살린다.
즉 S label은 하나의 subject-threshold polarity가 아니라, 최소 두 개의 sign regime을
row-target별로 배정해야 하는 문제에 가깝다.

다만 local 기대치인 `0.554752`까지는 가지 못했다. 따라서 union 안에는 public/private
assignment mismatch 또는 over-release cell이 섞여 있다. 후속 실험은 더 많은 cell을
추가하기보다, 이 360개 union 안에서 어떤 cell이 true S event이고 어떤 cell이 toxic
release인지 분리해야 한다.
