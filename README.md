# ETRI Competition - KBS HS-JEPA Public 0.55 Reproduction

이 브랜치는 public LB `0.5595724276`을 기록한 HS-JEPA Listener 실험을 팀원이 바로 이해하고 재현할 수 있도록 정리한 공유 패키지입니다.

핵심 폴더:

`hsjepa_public055/`

## 바로 실행

```bash
cd hsjepa_public055
python3 -m pip install -r requirements.txt
python3 src/reproduce_public055.py
```

성공하면 다음 파일이 생성됩니다.

`outputs/submission_public055_reproduced.csv`

이 파일은 제출 당시 public LB `0.5595724276`을 기록한 제출과 동일하게 재생산됩니다.

## 핵심 아이디어

S1/S2/S3/S4 수면지표를 하나의 sleep score에서 바로 나오는 값으로 보지 않고, 각 row-target이 서로 다른 hidden sign regime에 배정되는 문제로 보았습니다.

즉:

- 어떤 row-target은 subject-threshold 방향을 그대로 들어야 합니다.
- 어떤 row-target은 같은 신호를 반대 방향으로 들어야 합니다.
- 여러 해석기(subject-threshold, trajectory, measurement grammar, raw-semantic)가 같은 방향으로 동의할 때만 action을 release합니다.

이 two-regime row-target assignment가 public 0.55 진입의 핵심입니다.

자세한 설명:

`hsjepa_public055/docs/public055_experiment_explained_ko.md`

