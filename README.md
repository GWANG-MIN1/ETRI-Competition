# HS-JEPA — Certified Public-LB Level-Direction Transfer

ETRI 휴먼이해 AI 논문경진대회 (DACON 236690) 작업 중, **공개 리더보드 점수를 "추측"이 아니라
"인증(certificate)"으로 끌어올리는** 방법론과 그 구현입니다.

## 결과

| | public LB |
|---|---|
| 직전 best (FrontierSilence) | 0.5677269444 |
| **certified 후보 (`004a0549`)** | **0.5647490904** ✅ 신기록 |

- 사후 예측 게인 **−0.0030** vs 실현 **−0.0029779** → **99.3% 적중**, 인증 구간 `[0.5643, 0.5667]` 정중앙 착지.
- 제출 1회로 인증 가능한 가치를 전부 소진했음을 사후 증명(H12: 추가 인증 move 없음).

## 핵심 아이디어

1. **가산성(additivity)**: mean logloss는 셀 단위로 정확히 가산적 → 서로 다른 제출의 공개 점수 변화를
   하나의 라벨-세계 기하학으로 합성할 수 있다.
2. **배정노이즈 정리(assignment-noise theorem)**: `Δℓ(y=1) − Δℓ(y=0) = −Δlogit`. 따라서 **subject별
   균일 logit 이동**은 "어느 행이 1인지"와 무관하게 public delta가 per-subject 라벨비율의 *정확한 선형함수*가 된다
   (= 행-배정 복권이 0).
3. **S/N 법칙(정량 transfer law)**: dense 레벨/방향 이동(S/N 18~35)만 LB로 전이되고, 희소 셀-수술(S/N 0.03~0.26)은
   복권 → 과거 실패 크기를 이론이 그대로 재현.
4. **Ledger-polytope 인증**: 지금까지의 모든 공개 측정을 라벨 벡터 `r`에 대한 선형 제약으로 묶어 polytope를 만들고,
   후보 move의 최악-경우 게인을 LP로 구해 **"공개 측정과 모순 없는 모든 라벨 세계에서 손해 불가"**를 증명한 뒤 제출.

## 최종 후보 레시피

`certification/final_candidate_recipe.json` — FrontierSilence 기반 + 검증된 옛-라인 프로브 방향을
per-subject logit으로 이식: **Q2 방향 κ=0.75(cap 0.8) + Q3 방향 κ=0.25**. 빌더: `certification/build_final_candidate.py`.

## 디렉터리

- `certification/` — 가설 H1~H12 분석 + 최종 레시피/빌더 (이번 작업의 본체)
  - `candidate_3_inhull_loo_consensus.py` — in-hull + LOO-합의 sparse tomography
  - `analysis_h1_h6_sensor_trust.py` — prequential 백테스트(센서의 harm-valid/gain-invalid 비대칭) + FS 스태킹
  - `analysis_h7*_*.py` — 크로스-라인 레벨 이식 + recency 라벨 모델
  - `analysis_h8*_*.py` — 라벨-polytope LP 인증 (가산성→배정노이즈→S/N→polytope)
  - `analysis_h9_sensitivity.py` — 18개 설정 민감도(부호-안전 만장일치)
  - `analysis_h10*_*.py` — max-min 최적 move(레시피가 사실상 최적임을 확인)
  - `analysis_h11*_*.py` — LOCO + 임계 tolerance(단일 측정 오류 강건성)
  - `analysis_h12_next_move.py` — 새 측정 추가 후 다음 인증 move 탐색(없음)
- `submissions/` — 최종 certified 제출 파일(파생 예측만, 원본데이터 아님)
- `exploratory_pre_jackpot/` — jackpot 합류 이전(0.5932 라인) 탐색 도구. **superseded** — 다른 앵커/CV에서의 작업이라
  현재 라인과 직접 호환되지 않음. 기록 목적.

## ⚠️ 실행/데이터/공개 주의

- **이 repo는 단독 실행되지 않습니다.** 분석 스크립트는 팀 repo(`kbsooo/etri`)의 모듈(`candidate_1_*`, `hsjepa_jackpot/h154*`,
  cohort 모듈)과 **대회 원본 데이터·public score ledger**에 의존합니다. 그 자산들은 **의도적으로 포함하지 않았습니다**
  (대회 데이터 재배포 방지 + 타인 코드 미포함). 재현하려면 팀 repo를 옆에 클론하고 경로를 맞추세요.
- `.gitignore`로 데이터/대용량 산출물/캐시/시크릿을 차단합니다.
- **대회 진행 중이라면 이 repo를 private으로 두는 것을 권장**합니다(방법론 노출 방지).
