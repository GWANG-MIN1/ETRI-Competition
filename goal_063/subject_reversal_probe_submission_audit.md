# Subject Reversal Probe Submissions

## WARNING

MEMORY의 `GOAL_062` 확인 후 이 파일들은 **최종 후보가 아니라 위험한 진단 probe**로 강등한다.
결측/생활리듬은 base 오류 크기를 예측하지만 교정 방향은 안정적으로 찾지 못했고,
`GOAL_062` transfer-gate는 `PASS_LEVEL NONE`이었다. anchor 대체용으로 사용하지 않는다.

These files are diagnostic public probes, not final replacements for the 0.5615 anchor.

```text
                                         name                                                                                                                  path     hash changed_targets  mean_abs_delta  max_abs_delta  changed_cells  min_prob  max_prob
    submission_probe_subject_reversal_q3_m0p2     C:\Users\박광민\Documents\Codex\2026-06-18\handoff6-md-0-54-lb\outputs\submission_probe_subject_reversal_q3_m0p2.csv d94c8617              Q3        0.002015       0.025143            250  0.000003  0.999998
  submission_probe_subject_reversal_q123_m0p2   C:\Users\박광민\Documents\Codex\2026-06-18\handoff6-md-0-54-lb\outputs\submission_probe_subject_reversal_q123_m0p2.csv 82ebd4bd        Q1,Q2,Q3        0.005114       0.041209            750  0.000003  0.999998
submission_probe_subject_reversal_q123s4_m0p4 C:\Users\박광민\Documents\Codex\2026-06-18\handoff6-md-0-54-lb\outputs\submission_probe_subject_reversal_q123s4_m0p4.csv 08d1b525     Q1,Q2,Q3,S4        0.013112       0.082417           1000  0.051153  0.999998
```

## Residual Move By Subject

```text
subject_id        Q1        Q2        Q3        S1        S2        S3        S4
      id01 -0.012600  0.005181 -0.003620 -0.009110  0.010888  0.120367 -0.007921
      id02  0.017183  0.094154  0.102643  0.038327  0.062947  0.091804  0.010891
      id03  0.132478  0.206043  0.110496  0.016062 -0.036003 -0.061952 -0.127873
      id04 -0.029454 -0.053430  0.125717 -0.017187 -0.003589 -0.031479  0.028699
      id05  0.016242 -0.089870  0.074269 -0.147933 -0.146569 -0.201712 -0.069552
      id06 -0.155719 -0.076594 -0.050846  0.095258  0.057023  0.099448  0.076374
      id07 -0.004276 -0.012668 -0.011995  0.028399 -0.008294  0.050049 -0.065616
      id08 -0.013191  0.145854 -0.050523 -0.047451 -0.011313 -0.000580  0.034976
      id09  0.004410 -0.037534 -0.117015 -0.064573  0.047018 -0.009245 -0.014965
      id10  0.035518 -0.021850  0.057976 -0.102817 -0.119785 -0.105494  0.102884
```

## Usage

- Keep `submission_overshoot_x0p8_330ef1a1_uploadsafe.csv` as the protected final anchor.
- If using a public probe, start with `submission_probe_subject_reversal_q3_m0p2.csv` or `submission_probe_subject_reversal_q123_m0p2.csv`.
- Treat `q123s4_m0p4` as aggressive information gathering only.
