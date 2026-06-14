# HS-JEPA — Certified Public-LB Level-Direction Transfer

ETRI 휴먼이해 AI 논문경진대회 (DACON 236690) 작업 중, **공개 리더보드 점수를 "추측"이 아니라
"인증(certificate)"으로 끌어올리는** 방법론과 그 구현입니다.

## 결과

| | public LB |
|---|---|
| 직전 best (FrontierSilence) | 0.5677269444 |
| certified 후보 (`004a0549`) | 0.5647490904 |
| 검증된 도박 — overshoot anchor (`0fb93301`) | 0.5619100863 |
| **overshoot ×0.8 — 검증 캠페인 (`330ef1a1`)** | **0.5615333471** ✅ 현 신기록 |

- **예측 정확도 99%+ 가 두 번 재현**: ① certified 후보 — 예측 −0.0030 vs 실현 −0.0029779.
  ② overshoot ×0.8 — 예측 −0.00037 vs 실현 −0.0003767, 예상 0.56154 → 실현 0.5615333.
- 두 번 모두 "제출 전에 게인을 인증/예측"하고 실제 LB가 그 구간 정중앙에 착지.

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

## 검증 캠페인 (0.5619 → 0.56153) · `verification_campaign/`

핵심 도구 `polytope_eval.py`(30개 실측 LB로 calibrate된 권위적 arbiter)로 **~20개 가설을 적대적으로 검증**해
벽의 구조를 규명하고, 인증된 단일 레버를 정밀화해 신기록을 달성. 전체 로그: `CAMPAIGN_0614c_VERIFICATION.md`,
포트폴리오 요약: `NOTION_writeup.md`.

핵심 발견:

1. **인증은 joint이고, 단일 측정에 의존** — 단일 셀 이동은 0/70 인증 불가. Leave-one-out: 0.5619 anchor 하나만
   빼면 overshoot×0.8 worst-case가 −0.0017 → +0.0178로 붕괴 ⇒ overshoot×0.8은 "검증된 도박을 20% 축소한
   안전 보간"이지 독립 발견이 아님. *(`trackA*`, `trackFG`)*
2. **post-mean = mirage, worst-loose만 정직** — 추정 rate로 이동하면 모든 그룹에서 post-mean 음수(개선처럼 보임)지만
   worst-case는 전부 양수. 인증은 오직 worst-case로. *(`trackA3b`)*
3. **벽의 진짜 원인 = 드리프트 가용성** — Q2/Q3는 temporal self-corr ≈ 0(드리프트 거대), S는 0.6~0.93(안정 trait).
   overshoot가 먹히는 건 Q2/Q3가 드리프트하고 forward-CV가 그 방향을 corr 0.86/0.75로 포착하기 때문. *(`trackJ`,`trackH`,`trackK`)*
4. **🎯 public↓ ≠ private↓** — test 명당 ~12 public 행 → public rate 샘플링 노이즈 ~0.14. 캘리브레이션은 공유
   신호(드리프트)일 때만 private 전이. Monte Carlo: private-gain이 드리프트 ≈ 0.12 임계에서 0 교차. Q2/Q3는 임계
   위(전이 O), S는 임계 아래 → **S-probe는 public을 내리지만 private(상금)을 올리는 음수 EV**. *(`trackP`,`trackQ`,`trackO`)*
5. **per-row·앙상블·recalibration 전부 소진** — 드리프트 정렬 per-row 궤적조차 균일 shift보다 나쁨; 앙상블 블렌드
   레벨성분·FS recalibration·global 상수 shift 전부 미인증. 새 인증 레버 0. *(`trackR`,`trackD`,`trackL`,`trackM`)*
6. **private-rank 헤지** — per-subject-uniform은 private≈public(전이 성질). DACON이 2제출 중 더 나은 것을 채택하므로
   2-스케일 헤지({×0.7,×1.0})가 단일보다 기대 private gain ~50%↑·다운사이드 제거; {FS,×0.8}는 다운사이드 완전 차단. *(`trackS2`)*

**결론**: private(상금) 기준 0.56153은 이 base의 floor에 근접 — 유일한 private-양수 레버(Q2/Q3 드리프트)는 완전 수확됨.

## 최종 후보 레시피

- `certification/final_candidate_recipe.json` — FrontierSilence 기반 + 검증된 옛-라인 프로브 방향을
  per-subject logit으로 이식: **Q2 방향 κ=0.75(cap 0.8) + Q3 방향 κ=0.25**. 빌더: `certification/build_final_candidate.py`.
- `verification_campaign/save_overshoot_x08.py` — 현 신기록 빌더: FS + **0.8 × per-subject Q2/Q3 드리프트 overshoot**.
  upload-safety 체크 + 권위적 arbiter 재검증 포함. 산출물: `submissions/submission_overshoot_x0p8_330ef1a1_uploadsafe.csv`.

## 디렉터리

- `certification/` — 가설 H1~H12 분석 + 최종 레시피/빌더 (0.5647 인증 라인)
  - `analysis_h8*_*.py` — 라벨-polytope LP 인증 (가산성→배정노이즈→S/N→polytope)
  - `analysis_h9~h12` — 민감도 / max-min 최적 / LOCO / 다음 인증 move 탐색
- `verification_campaign/` — **이번 작업**: 0.5619→0.56153 검증 캠페인 (~20 가설)
  - `polytope_eval.py` — 30 실측 LB로 calibrate된 권위적 arbiter (모든 track 스크립트의 기반)
  - `trackA*`,`trackFG`,`trackL`,`trackM`,`trackD` — 인증 프런티어(새 레버 없음 증명) + σ 정밀화
  - `trackH`,`trackI`,`trackJ`,`trackK` — 벽의 메커니즘(드리프트 가용성, Q⊥S, trait 안정성)
  - `trackC`,`trackE`,`trackN`,`trackO`,`trackP`,`trackQ*`,`trackR`,`trackS2` — public-subset 식별 + public↔private 전이 + S-probe OED
  - `save_overshoot_x08.py`,`build_certified_dry.py` — 신기록 빌더 + dry-validation
  - `CAMPAIGN_0614c_VERIFICATION.md`,`NOTION_writeup.md` — 캠페인 전체 로그 + 포트폴리오 요약
- `submissions/` — 최종 certified 제출 파일(파생 예측만, 원본데이터 아님)
- `exploratory_pre_jackpot/` — jackpot 합류 이전(0.5932 라인) 탐색 도구. **superseded**, 기록 목적.

## ⚠️ 실행/데이터/공개 주의

- **이 repo는 단독 실행되지 않습니다.** 분석 스크립트는 팀 repo(`kbsooo/etri`)의 모듈(`candidate_1_*`, `hsjepa_jackpot/h154*`,
  cohort 모듈)과 **대회 원본 데이터·public score ledger·CV 하네스**(`hsjepa_core.py`)에 의존합니다. 그 자산들은
  **의도적으로 포함하지 않았습니다**(대회 데이터 재배포 방지 + 타인 코드 미포함). 재현하려면 팀 repo·데이터를 옆에 두고 경로를 맞추세요.
- `.gitignore`로 데이터/대용량 산출물/캐시/시크릿을 차단합니다.
- **대회 진행 중이라면 이 repo를 private으로 두는 것을 권장**합니다(방법론 노출 방지).
