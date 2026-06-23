# HS-JEPA Public 0.55 Reproduction Package

## 무엇을 재현하나

이 패키지는 ETRI 수면 기반 생활습관 로그 예측 대회에서 public LB `0.5595724276`을 기록한 제출을 재현합니다.

제출 당시 파일명:

`submission_human_life_s_hidden_generator_council_solver_two_regime_assignment_union_k360_dc3a5f9d53_uploadsafe.csv`

이 패키지에서 재생성되는 파일:

`outputs/submission_public055_reproduced.csv`

## 실행 방법

```bash
python3 -m pip install -r requirements.txt
python3 src/reproduce_public055.py
```

정상 실행 시 다음이 출력됩니다.

- expected public LB: `0.5595724276`
- stable target hash: `dc3a5f9d53`
- expected submission과의 최대 차이: `~1e-16`
- S1/S2/S3/S4별로 움직인 cell 수

## 폴더 구조

```text
hsjepa_public055/
  data/
    ch2026_submission_sample.csv
    conflict_rescue_base_public_0p5612880941.csv
    two_regime_assignment_union_k360_selected_cells.csv
    candidate_summary.csv
  docs/
    public055_experiment_explained_ko.md
    original_solver_note_ko.md
  outputs/
    submission_public055_reproduced_expected.csv
  src/
    reproduce_public055.py
  requirements.txt
```

## 핵심 데이터 파일 설명

`conflict_rescue_base_public_0p5612880941.csv`

직전 기준 제출입니다. public LB `0.5612880941`을 기록했습니다.

`two_regime_assignment_union_k360_selected_cells.csv`

최종 제출로 이동시킨 360개 S row-target cell 목록입니다. 각 row-target에 대해 어떤 target을 얼마나 logit 이동할지 들어 있습니다.

`submission_public055_reproduced_expected.csv`

실제 public LB `0.5595724276`을 기록한 제출 파일의 expected copy입니다. 재현 스크립트는 새로 만든 파일이 이 파일과 같은지 검증합니다.

## 아이디어 요약

S1/S2/S3/S4는 objective sleep metric처럼 보이지만, 실험 결과 raw sleep state의 단순 threshold로는 잘 설명되지 않았습니다.

이번 실험은 S label을 다음 문제로 다시 정의했습니다.

> 각 row-target은 normal sign regime 또는 inverse sign regime 중 하나에 속한다.
> 예측의 핵심은 좋은 sleep score 하나를 만드는 것이 아니라, row-target별 hidden sign regime을 배정하는 것이다.

그래서 두 종류의 cell을 합쳤습니다.

- Regime A: subject-threshold 방향을 그대로 믿는 180개 cell
- Regime B: subject-threshold 방향을 뒤집었을 때 더 잘 들리는 180개 cell

둘을 합친 360개 cell이 `two_regime_assignment_union_k360`입니다.

## 문서

팀원에게 개념부터 설명하려면 먼저 이 문서를 보면 됩니다.

`docs/public055_experiment_explained_ko.md`

기존 실험 로그에 가까운 원본 노트는 다음입니다.

`docs/original_solver_note_ko.md`

## 주의

이 패키지는 거대한 전체 실험 트리를 그대로 옮긴 것이 아닙니다.

목표는 팀원이 0.55 제출을 정확히 재현하고, 그 제출이 어떤 세계관과 알고리즘에서 나온 것인지 이해하는 것입니다.

