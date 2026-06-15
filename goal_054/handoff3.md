# HS-JEPA / DACON #236690 — Handoff 3 (2026-06-14~15 세션)

> 작성일: 2026-06-15 · 대회: 제5회 ETRI 휴먼이해 AI 논문경진대회 (수면/스트레스, **논문** 대회)
> 한 줄 요약: **overshoot×0.8 제출 → public LB 0.5615333471 달성(예측 99% 적중). 이후 "0.54 도전" 캠페인: 정보이론 floor·reconstruction·cross-target 전부 데이터한계로 막힘 → 상위팀 방법(MIS-LSTM, arXiv:2509.11232) 발견·CPU검증(딥-시퀀스가 Q2에서 honest −0.035 직교신호). 단 450행 데이터한계로 모델 스케일업=과적합(GPU 무익). 최선=작은 v3 모델(CPU). 딥이 FS를 이기는지는 LB로만 판정 가능 → 보수적 블렌드 후보 제출 대기.**
> 직전 handoff2.md(0.5619 앵커 시점) 이후 전 진전. 메인 로그: 메모리 `experiment_log_0614c.md`.

---

## 0. 대회 / 메트릭 (불변)
- **메트릭**: 250 test rows × 7 binary targets 평균 logloss. **낮을수록 좋음.** public LB = test의 ~50% 서브셋(~125행, 명당 ~12행).
- **7 타깃**: Q1/Q2/Q3 = 주관 웰빙(개인평균 대비 위/아래, ~50% 기저율). **S1=TST/S2=SE/S3=SOL/S4=WASO = 객관 NSF**(Withings 침대센서 측정값에 가이드라인 임계 적용 = 결정론적).
- **구조**: test = train과 동일 10명. 62.4% interleaved / 37.6% future. train 450 rows(~45/subj). **logloss는 calibration-bound.**
- **공식 메트릭 정의**: `data/ch2026_metrics_description.pdf`. 대회 기반 데이터 = 공개된 **ETRI Lifelog Dataset 2024**(nanum.etri.re.kr, arXiv:2508.03698). 우리가 받은 모달리티 12개(폰/워치)는 공개셋과 동일. **Withings 침대센서 raw는 미제공**(그래서 S가 예측 과제).

---

## 1. 현재 상태 (best & 제출)

| 항목 | 값 |
|---|---|
| **public LB best (제출됨)** | **0.5615333471** |
| best 정체 | **overshoot×0.8** (FS + 0.8×per-subject Q2/Q3 드리프트 overshoot, hash `330ef1a1`) |
| 빌더 | `save_overshoot_x08.py` → `submission_overshoot_x0p8_330ef1a1_uploadsafe.csv` |
| 예측 검증 | 예측 −0.00037 vs 실현 −0.0003767 (**99%+ 적중**, anchor 0.5619100863 대비) |
| 이전 anchor | 0.5619100863 (overshoot×1.0) |
| FS base | `submission_hsjepa_..._1e013277_uploadsafe.csv` (0.5677269444) |
| 2점 곡선fit | σ*=0.80이 이 축의 **floor**(σ=0.75→0.561559, σ=0.85→0.561555 모두 약간 나쁨) |
| GitHub | `GWANG-MIN1/etri` main에 0614c 검증캠페인 7커밋 푸시(verification_campaign/ + submissions/) |

> ⚠️ 리더보드 상황(사용자 제공): **현재 15등, 상위 5등 = 0.54.** 즉 0.54는 (적어도 public에선) 도달 가능 = 신호 존재.

---

## 2. 🎯 0.54 캠페인 — 핵심 발견 (의사결정 근거)

### 2.1 정보이론 floor (goal_floor_analysis.py)
- per-subject 캘리브레이션 바닥(per-row 정보 0) = **0.6182**. 현재 0.5615는 그보다 **0.057 아래** = jackpot이 이미 상당한 per-row 판별 추출중.
- **0.54는 캘리브바닥보다 0.078 아래** → 레벨/캘리브로 불가, **전이되는 per-row 판별이 추가로 필요**(jackpot 점프 0.025 규모를 한 번 더).
- public-overfit floor(per-subject 캘리브를 ~12행 public에 fit) = **0.5613 ≈ 현재**. → 0.54는 단순 과적합으로도 불가, **진짜 per-row 신호** 필요.

### 2.2 닫힌 경로들 (데이터 한계 = 450행/10명에서 뭐든 과적합)
- **객관 S 재구성**(폰 수면창→TST/SE/SOL/WASO→NSF임계): 폰 프록시 AUC 0.47–0.67 vs **팀 모델 0.74–0.78** → subsumed. 팀 OOF에 직교 추가 시 **4/4 S 악화**. (팀이 이미 S를 잘 함.)
- **cross-target 스태킹**(7타깃 예측+재구성을 서로 피처로): 전 타깃 악화(과적합).
- **7-패러다임 적대평가**(workflow `paradigm-search-054`): FM임베딩·외부수면전이·joint·transductive·JEPA 등 전부 DEAD/likely-dead. (단 일반 인코더 가정; MIS-LSTM은 별개로 후속 발견.)
- 야간 HR 커버리지: 워치는 야간 21%만(폰은 97%).

### 2.3 ⭐ 패러다임 발견: MIS-LSTM (arXiv:2509.11232)
**바로 이 대회("450일 학습/250 held-out/10명")의 발표된 상위 방법.** 외부데이터 없음(합법·private 전이).
- **구조**: 분당 다채널 센서 시퀀스 → 다채널 **이미지** → **SEResNeXt101 백본** → CBAM → **2층 LSTM** → **subject embedding(개인화)** → 7출력. focal loss, **UALRE 앙상블**.
- **성능**: Macro-F1 0.615(단일)/0.647(앙상블). baseline LSTM 0.576/CNN 0.578.
- **함의**: 팀의 per-row 천장(0.5677)은 CatBoost-on-daily-aggregate + 실패한 tiny-seqcnn(1225p)에서 나옴. **놓친 건 모델 클래스**(딥-시퀀스+subject-emb). seqcnn이 DEAD였던 건 모델이 너무 작아서지 신호가 없어서가 아님.

### 2.4 CPU 검증 (이 모델 클래스의 실측)
- 경량 Conv+BiLSTM+subject-emb, **13채널**(HR mean/std·steps·screen·charge·still·light + GPS속도·usage·wifi·ble·ambience-speech·wLight).
- **honest nested 검증**(가중치 한쪽절반서 선택→반대쪽 적용): **Q2 −0.0354(견고!)**·Q3 −0.0069 = 딥-시퀀스가 **주간 활동/사회맥락으로 취침전 피로/스트레스를 잡는** 직교 per-row 신호(팀 daily-aggregate가 놓침). 단 Q1+0.0096·S1+0.0225 = 과적합(딥은 **Q2/Q3에만** 적용).
- 단 게인은 **unified OOF(0.588) 대비** 검증. **FS(0.5615)와의 블렌드는 FS-OOF 부재로 사전검증 불가.**

### 2.5 🚨 GPU/스케일 = 무익 (데이터 한계 재확인)
- Colab GPU 실행(HID192·150ep): **과적합** (Q2 OOF 0.711, Q1/Q3는 기저율보다도 나쁨).
- over-수정(HID64·dropout0.55·WD3e-3): **과소적합** (Q2 0.678).
- **스위트스팟 = v3(HID96·dropout0.4·무증강·80ep) → Q2 OOF 0.648** = CPU에서 이미 돌아감.
- ⇒ **병목은 컴퓨팅이 아니라 데이터(450행). GPU로 모델 키우면 과적합. (내 Conv+BiLSTM 클래스 한정; 논문 SEResNeXt-이미지는 별개·미검증.)** "GPU 필요"는 정정됨.

---

## 3. 제출 후보 & 결정

| 후보 | 파일 | 블렌드(deep vs FS) | 상태 |
|---|---|---|---|
| **best(제출됨)** | `submission_overshoot_x0p8_330ef1a1_uploadsafe.csv` | (overshoot×0.8) | **LB 0.5615333471** |
| MIS-LSTM 보수(권장 1차) | `submission_mislstm_conservative.csv` | Q2 0.2·Q3 0.12·S3 0.05 | 제출 대기 |
| MIS-LSTM 공격 | `submission_mislstm_full.csv` | Q2 0.4·Q3 0.25 | 제출 대기 |

- **유일한 판정법 = LB**(FS-OOF 없음). 보수적 후보 먼저 제출 → 0.5615보다 내려가면 딥 신호가 FS 도움(가중치↑), 올라가면 FS가 포섭(0.5615 유지).
- **솔직한 전망**: 딥 모델은 가장 강한 합법 per-row 레버지만 데이터한계로 Q2 modest 게인. 단독 0.54는 어려움. 상위팀 0.54가 private에서도 진짜면 더 정교한 레시피 or public 과적합 요소.

---

## 4. 파일 인벤토리 (이번 세션 산출, `final_hsjepa_candidates/`)
- **제출/빌더**: `save_overshoot_x08.py`, `build_certified_dry.py`, `submission_overshoot_x0p8_*.csv`, `submission_mislstm_{full,conservative}.csv`
- **0614c 검증캠페인 도구**: `outputs/track{A,A2,A3b,C,D,E,FG,H,I,J,K,L,M,N,O,P,Q,Q2,R,S2}*.py`, `polytope_eval.py`, `CAMPAIGN_0614c_VERIFICATION.md`, `NOTION_writeup.md`
- **0.54 캠페인 도구**: `outputs/goal_{floor_analysis,night_coverage,sleep_reconstruct,sleep_reconstruct_v2,recon_orthogonality,public_overfit_reachability,crosstarget_stack,mislstm_prep,mislstm_train,mislstm_v2,mislstm_prep_v3,mislstm_train_v3,mislstm_antiof_check}.py`
- **MIS-LSTM 배포**: `outputs/mislstm_full.py`(turnkey 13ch+test+FS블렌드+제출), `outputs/mislstm_colab.py`+`outputs/MIS_LSTM_colab.ipynb`(Colab GPU판), `outputs/mislstm_data{,_v3}.npz`(캐시 시퀀스), `etri_team/etri_colab_data.zip`(Colab용 122MB 데이터)
- **인프라**: CV 하네스 `2026-05-26/.../src/hsjepa_core.py`; venv `2026-05-26/.../.venv`(torch CPU전용·4코어); raw parquet `etri_team/data/ch2025_data_items/`; ledger `etri_team/data_analytics/hsjepa_public_score_ledger.csv`
- ⚠️ Win 주의: 스크립트에서 **torch를 numpy/pandas보다 먼저 import**(MKL DLL 충돌). 백그라운드 실행도 import 순서만 맞으면 OK.

---

## 5. 다음 작업 (Pending)
1. **보수적 MIS-LSTM 후보 제출** → public LB 확인. 내려가면 가중치↑/공격버전, 올라가면 0.5615 방어.
2. (선택) 논문 정확 재현: SEResNeXt101 per-block 이미지 + CBAM(GPU). 단 데이터한계로 강한 정규화 필수, 효과 불확실.
3. (선택) overshoot×0.8를 DACON 최종 2제출 헤지로: {×0.8, FS} 또는 {×0.7, ×1.0}(trackS2: best-of-2가 단일보다 우월, CVaR↓).
4. (논문각) human-state 발견: Q2/Q3=주간활동/사회맥락 시퀀스로 예측가능(딥-시퀀스 직교신호), S=객관 stable trait.

## 6. 메모리 포인터
- `experiment_log_0614c.md` ⭐⭐⭐ — 이번 세션 전 로그(0614c 검증캠페인 18가설 + 0.54 재오픈 + reconstruction + MIS-LSTM 발견·검증).
- `experiment_log_0614.md`, `experiment_log_0612.md` — 직전(0.5647 천장, polytope/gate 도구, jackpot 인프라).
- `dacon_submission_checklist.md` — 제출 전 7단계 검증.
