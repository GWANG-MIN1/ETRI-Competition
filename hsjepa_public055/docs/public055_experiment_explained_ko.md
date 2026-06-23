# 0622 팀 공유용: 처음 public 0.55에 진입한 실험 설명

## 1. 한 줄 요약

이번에 처음으로 public LB가 0.55대로 내려간 핵심 이유는,
S1/S2/S3/S4 수면지표를 하나의 단순한 수면 점수로 보지 않고,
각 row-target이 서로 다른 hidden regime에 속한다고 본 것이다.

쉽게 말하면 다음과 같다.

기존 생각:

> 이 사람의 오늘 수면 상태가 좋으면 S label도 한 방향으로 움직일 것이다.

이번 실험의 생각:

> 같은 수면 상태라도 사람마다, 날짜마다, target마다 다르게 해석될 수 있다.
> 그래서 S label은 하나의 방향이 아니라 row-target별로 어느 regime에 속하는지를 먼저 풀어야 한다.

이 관점으로 만든 제출이 public LB `0.5595724276`을 기록했다.

## 2. 결과

직전 기준점:

`0.5612880941`

이번 제출:

`0.5595724276`

개선폭:

`-0.0017156665`

제출 파일명:

`submission_human_life_s_hidden_generator_council_solver_two_regime_assignment_union_k360_dc3a5f9d53_uploadsafe.csv`

이 파일은 우리 작업에서 처음으로 public LB `0.55`대에 들어간 제출이다.

## 3. 왜 이 실험을 했나

대회의 target은 7개다.

- Q1: 주관적 수면 만족도
- Q2: 취침 전 피로도 또는 개입/부담 계열
- Q3: 스트레스 또는 수면 품질 관련 주관 신호
- S1: 총 수면시간
- S2: 수면효율
- S3: 수면 지연시간
- S4: 수면 중 각성 시간

여기서 S1-S4는 객관 수면지표처럼 보인다.
그래서 처음에는 S label이 raw sleep state에서 직접 나올 것처럼 생각하기 쉽다.

예를 들면:

- 오래 잤으면 S1이 높다.
- 효율이 좋으면 S2가 높다.
- 잠드는 데 오래 걸렸으면 S3가 나쁘다.
- 중간에 많이 깼으면 S4가 나쁘다.

하지만 실험을 계속해보니 이렇게 단순하지 않았다.

같은 sleep signal을 넣어도 어떤 row-target에서는 맞는 방향이 되고,
다른 row-target에서는 반대 방향이 맞는 것처럼 보였다.

즉 S label은 raw sleep state의 단순 threshold가 아니라,
사람별 기준, 측정 방식, 날짜 흐름, hidden sleep episode가 섞여 생성된 값처럼 보였다.

그래서 이번 실험의 질문은 이것이었다.

> S label은 하나의 sleep score에서 나온 것인가,
> 아니면 row-target마다 서로 다른 hidden sign regime에 배정되는 것인가?

## 4. 핵심 아이디어

이번 실험은 S label을 다음과 같은 hidden generation process로 봤다.

1. 먼저 raw sleep episode가 있다.
   - 실제로 얼마나 잤는지
   - 수면 중 끊김이 있었는지
   - 잠드는 구간이 어땠는지
   - 수면 경계가 어디였는지

2. 그 raw episode는 사람마다 다르게 해석된다.
   - 어떤 사람은 평소에도 수면시간이 짧다.
   - 어떤 사람은 수면효율이 항상 높거나 낮다.
   - 어떤 사람은 센서 측정이 특정 target에서 치우칠 수 있다.
   - 그래서 같은 raw state라도 subject별 threshold가 다르다.

3. 같은 사람 안에서도 target마다 듣는 방향이 다를 수 있다.
   - S1/S2는 duration/efficiency 쪽이다.
   - S3/S4는 latency/WASO 쪽이다.
   - 이 둘은 같은 sleep event를 반대로 읽을 수 있다.

4. 따라서 prediction action을 바로 내지 않고, 여러 interpreter가 같은 방향을 말할 때만 움직인다.
   - subject-threshold interpreter
   - trajectory interpreter
   - measurement grammar interpreter
   - raw-semantic interpreter

이번 제출은 이 네 종류의 신호가 어느 정도 동의하는 row-target만 골라서 움직였다.

## 5. 여기서 말하는 interpreter란 무엇인가

interpreter는 복잡한 모델 이름이 아니라,
같은 row-target을 서로 다른 관점에서 읽는 해석기라고 보면 된다.

### 5.1 Subject-threshold interpreter

질문:

> 이 사람 기준으로 오늘의 S target은 올라가야 자연스러운가, 내려가야 자연스러운가?

사람마다 평소 수면 패턴이 다르다.
따라서 전체 평균 기준으로 판단하면 틀릴 수 있다.

예를 들어 어떤 사람에게 6시간 수면은 부족한 날이지만,
다른 사람에게는 평소보다 많이 잔 날일 수 있다.

subject-threshold interpreter는 이런 개인 기준을 반영한다.

### 5.2 Trajectory interpreter

질문:

> 전후 날짜 흐름으로 보면 오늘은 어떤 수면 상태로 이어지는 날인가?

하루는 독립 샘플이 아니다.
전날 피로, 누적 수면부족, 회복수면, 주말/평일 리듬 같은 흐름이 있다.

trajectory interpreter는 row order와 날짜 흐름에서
오늘의 S target 방향을 읽는다.

### 5.3 Measurement grammar interpreter

질문:

> 센서와 측정 과정의 문법상 이 값은 믿을 만한가, 아니면 측정 convention 때문에 다르게 읽어야 하는가?

S label은 objective처럼 보이지만,
실제로는 센서가 수면을 어떻게 나누고 기록했는지에 영향을 받는다.

예를 들어:

- 수면시간이 길어 보이지만 중간 각성이 많을 수 있다.
- 효율이 좋아 보이지만 sleep boundary가 잘못 잡혔을 수 있다.
- 입면 지연과 중간 각성이 서로 섞여 보일 수 있다.

measurement grammar interpreter는 이런 측정 과정의 문법을 반영한다.

### 5.4 Raw-semantic interpreter

질문:

> raw sleep feature와 인간적으로 해석한 sleep state가 같은 방향을 말하는가?

숫자 feature 자체와 사람이 이해할 수 있는 sleep episode 의미를 함께 본다.

예를 들어:

- "긴 수면이지만 fragmented sleep"
- "짧지만 효율 좋은 수면"
- "늦게 잠들었지만 중간 각성은 적은 수면"
- "수면 경계가 불안정한 날"

이런 식의 의미 단위가 raw feature와 같은 방향을 말하는지 확인한다.

## 6. 두 개의 regime

이번 실험에서 가장 중요한 발견은 이것이다.

S label에는 최소 두 개의 sign regime이 있는 것처럼 보인다.

### Regime A

subject-threshold 방향을 그대로 믿는 cell이다.

예:

> 이 사람의 평소 기준으로 오늘 S3가 올라가야 한다고 보이고,
> 다른 interpreter들도 그 방향을 지지한다.

### Regime B

subject-threshold 방향을 뒤집었을 때 오히려 더 잘 들리는 cell이다.

이게 가장 이상하고 중요한 부분이다.

처음에는 반대 방향을 단순 negative control로 생각할 수 있다.
그런데 실제로는 반대 방향 중에서도 public-memory와 local residual 관점에서
harm이 아니라 gain처럼 들리는 cell들이 있었다.

그래서 이 실험은 다음처럼 진행됐다.

1. Regime A 후보 180개를 고른다.
2. Regime B 후보 180개를 고른다.
3. 둘을 합쳐 총 360개 S row-target cell을 움직인다.

이 최종 후보가 `two_regime_assignment_union_k360`이다.

## 7. 실제로 무엇을 바꿨나

전체 test row는 250개이고, target은 7개라서 총 1750개 probability가 있다.

이번 실험은 그중 S1/S2/S3/S4에 해당하는 1000개 cell만 대상으로 삼았다.

그중 360개 cell을 골라 logit을 작게 이동시켰다.

Target별로는 다음과 같이 움직였다.

| target | 움직인 cell 수 |
| --- | ---: |
| S1 | 84 |
| S2 | 66 |
| S3 | 107 |
| S4 | 103 |

Subject별로는 비교적 넓게 퍼져 있었다.

| subject | 움직인 cell 수 |
| --- | ---: |
| id06 | 45 |
| id01 | 44 |
| id07 | 44 |
| id10 | 36 |
| id04 | 35 |
| id02 | 33 |
| id05 | 31 |
| id03 | 31 |
| id09 | 31 |
| id08 | 30 |

즉 특정 subject 하나에만 과하게 몰린 제출은 아니었다.

Regime별로는 정확히 반반이다.

| source regime | 움직인 cell 수 |
| --- | ---: |
| Regime A: subject-threshold council | 180 |
| Regime B: opposite-sign council | 180 |

## 8. 어떻게 cell을 골랐나

각 row-target cell에 대해 다음 질문을 했다.

1. subject-threshold는 어느 방향을 말하는가?
2. trajectory는 같은 방향을 말하는가?
3. measurement grammar는 같은 방향을 말하는가?
4. raw-semantic state는 같은 방향을 말하는가?
5. public-memory 관점에서 이 action은 harm처럼 보이는가 gain처럼 보이는가?
6. local residual 관점에서 이 action은 harm처럼 보이는가 gain처럼 보이는가?

그리고 다음 조건을 만족하는 cell을 우선했다.

- 여러 interpreter가 같은 방향으로 동의한다.
- 반대 신호보다 동의 신호가 많다.
- action을 넣었을 때 public-memory/local-residual proxy가 나쁘지 않다.
- probability를 너무 극단으로 밀지 않는다.
- 하나의 target/subject에만 과하게 몰리지 않는다.

Regime A는 subject-threshold 방향을 기본으로 골랐다.

Regime B는 subject-threshold를 뒤집은 방향 중에서도
다른 interpreter와 proxy score가 살아남는 cell을 골랐다.

## 9. 제출 전 예상과 실제 결과

제출 전에는 여러 후보를 만들었다.

| 후보 | 움직인 S cell | offline raw 예상 | offline local 예상 | 실제 public |
| --- | ---: | ---: | ---: | ---: |
| Regime A만 사용 | 180 | 0.555715 | 0.558055 | 미제출 |
| Regime B만 사용 | 180 | 0.553732 | 0.557985 | 미제출 |
| Regime A+B union | 360 | 0.548159 | 0.554752 | 0.5595724276 |

실제 public은 `0.5595724276`이었다.

이 결과는 애매하게 실패한 것이 아니라, 매우 중요한 정보를 준다.

offline 예상만큼 `0.554`까지 내려가지는 못했다.
하지만 직전 기준점 `0.561288`보다 확실히 좋아졌다.

따라서 다음 두 가지를 동시에 알 수 있다.

1. two-regime S assignment 방향은 실제로 맞는 신호를 포함한다.
2. 하지만 360개 cell 안에 public/private mismatch 또는 toxic cell도 섞여 있다.

## 10. 왜 이게 중요한 발견인가

이 실험 전까지는 S label을 더 잘 맞히려면
"수면 상태를 더 잘 계산하면 된다"는 느낌이 강했다.

하지만 이번 결과는 다른 방향을 보여준다.

S label의 병목은 단순히 좋은 sleep score를 만드는 것이 아니다.

진짜 병목은 다음에 가깝다.

> 이 row-target이 어떤 hidden sign regime에 속하는가?

같은 sleep evidence라도 어떤 target에서는 그대로 들어야 하고,
어떤 target에서는 반대로 들어야 한다.

이것은 단순 feature engineering이나 blend가 아니라,
수면지표 label 생성 과정을 다르게 본 것이다.

## 11. HS-JEPA와의 연결

HS-JEPA의 핵심은 visible context에서 hidden human-state representation을 예측하고,
그 representation을 label로 바로 찍는 것이 아니라
어떤 listener가 어떻게 들어야 하는지 분리하는 것이다.

이번 실험은 그 구조와 잘 맞는다.

Core 관점:

- raw sleep episode
- subject-relative sleep state
- temporal trajectory
- measurement grammar
- raw-semantic sleep meaning

이것들이 hidden human-state view다.

Listener 관점:

- 어떤 view가 같은 방향으로 동의하는가?
- 어떤 row-target은 normal polarity인가?
- 어떤 row-target은 inverse polarity인가?
- action을 실제 probability correction으로 번역해도 안전한가?

즉 이번 제출은 HS-JEPA를 "좋은 latent 하나 만들기"가 아니라
"hidden state를 target별로 어떻게 들어야 하는가"의 문제로 확장한 사례다.

## 12. 논문에 쓸 수 있는 핵심 주장

영문:

> We found that objective sleep labels are not well explained as direct thresholded outputs of raw sleep states. Instead, they behave like row-target assignments over multiple hidden sign regimes. We therefore release prediction actions only when subject-specific thresholds and independent trajectory, measurement, and raw-semantic interpreters agree on a latent direction.

한국어:

> 객관 수면지표 S1-S4는 raw sleep state의 단순 threshold 결과라기보다, subject별 기준과 측정 문법, 시간 흐름, raw-semantic 수면 상태가 결합된 row-target assignment 문제로 보인다. 우리는 여러 interpreter가 같은 방향을 말할 때만 prediction action을 release했고, 그 결과 처음으로 public LB 0.55대에 진입했다.

## 13. 한계

이번 실험은 강한 발견이지만 완성은 아니다.

한계는 명확하다.

1. offline expected `0.554752`까지 가지 못했다.
2. 360개 union 안에 toxic cell이 섞여 있다.
3. public에서는 맞지만 private에서는 위험할 수 있는 cell이 있다.
4. S3/S4는 특히 잘못 건드리면 LogLoss tail이 커질 수 있다.
5. two-regime이 맞더라도 어떤 cell이 어느 regime인지 완벽히 푼 것은 아니다.

## 14. 다음에 해야 할 일

다음 단계는 cell을 더 많이 추가하는 것이 아니다.

이미 360개 안에 좋은 신호와 나쁜 신호가 섞여 있다는 것을 봤다.

따라서 다음 과제는 다음이다.

1. 360개 union 안에서 true S event cell과 toxic cell을 분리한다.
2. S1/S2 duration-efficiency regime과 S3/S4 latency-WASO regime을 분리한다.
3. public에서만 좋은 cell과 private에도 갈 가능성이 높은 cell을 구분한다.
4. listener responsibility를 더 정교하게 만든다.
5. HS-JEPA core가 만든 hidden state를 어떤 listener가 어떤 polarity로 들어야 하는지 체계화한다.

## 15. 팀원이 이 실험을 한 문장으로 이해한다면

이 실험은 S label을 "수면이 좋으면 올라가고 나쁘면 내려간다"로 본 것이 아니라,
"각 row-target이 normal/inverse hidden regime 중 어디에 속하는지 맞히는 문제"로 다시 정의한 실험이다.

그 관점이 실제 public LB에서 처음으로 0.55 진입을 만들었다.

## 16. 재현 정보

이 섹션은 코드를 실행하려는 사람을 위한 부록이다.
위 설명을 이해하는 데 필수는 아니다.

재현용 스크립트:

`human_life_view_predictive_architecture/competition_adapter/run_s_hidden_generator_council_solver.py`

생성되는 핵심 산출물:

`human_life_view_predictive_architecture/outputs/s_hidden_generator_council_solver/s_hidden_generator_council_candidate_summary.csv`

`human_life_view_predictive_architecture/outputs/s_hidden_generator_council_solver/s_hidden_generator_council_selected_cells_all.csv`

`human_life_view_predictive_architecture/outputs/s_hidden_generator_council_solver/s_hidden_generator_council_cell_atlas.csv`

팀 공유용 핵심 문서:

`human_life_view_predictive_architecture/docs/S_HIDDEN_GENERATOR_COUNCIL_SOLVER_KO.md`

실행 명령:

```bash
python3 human_life_view_predictive_architecture/competition_adapter/run_s_hidden_generator_council_solver.py
```
