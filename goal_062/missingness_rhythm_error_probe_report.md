# Missingness/Rhythm Error Probe

Question: can observed lifelog missingness and lifestyle rhythm identify who/when the current base is badly wrong?

## Data

- Train rows: `450`
- Test rows feature-aligned: `250`
- Raw/lifelog-derived numeric columns selected from feature store: `4951`
- Final model feature columns after cleaning/deviation expansion: `320`

## Error-Detector Quality

```text
          split model  mean_signed_corr  mean_high_error_auc  max_high_error_auc  mean_top20_loss_lift
    interleaved ridge         -0.000246             0.511735            0.615664              0.029096
subject_holdout ridge          0.033131             0.494813            0.531512             -0.001220
```

## Future-Tail Check

```text
model  mean_future_delta  mean_future_high_error_auc
ridge           0.021305                    0.480315
```

## Best OOF Candidate Families

```text
          split model              family   level  param  mean_delta  delta_Q1  delta_Q2  delta_Q3  delta_S1  delta_S2  delta_S3  delta_S4
    interleaved ridge              signed subject   0.80   -0.003604 -0.002569 -0.003422 -0.007102 -0.001649 -0.004475 -0.003279 -0.002730
    interleaved ridge              signed subject   0.50   -0.002766 -0.002021 -0.002701 -0.005189 -0.001651 -0.003207 -0.002227 -0.002364
    interleaved ridge              signed subject   0.30   -0.001861 -0.001377 -0.001836 -0.003407 -0.001246 -0.002083 -0.001406 -0.001675
    interleaved ridge              signed  block6   0.30   -0.001263  0.002797  0.001562 -0.001050 -0.000003 -0.004557 -0.004440 -0.003154
    interleaved ridge              signed subject   0.15   -0.001006 -0.000749 -0.000997 -0.001813 -0.000722 -0.001100 -0.000729 -0.000933
    interleaved ridge              signed  block6   0.15   -0.000908  0.001104  0.000528 -0.000863 -0.000274 -0.002586 -0.002416 -0.001851
    interleaved ridge              signed  block6   0.50   -0.000811  0.006079  0.003777 -0.000169  0.001280 -0.006193 -0.006498 -0.003952
    interleaved ridge risk_shrink_subject  block6   0.65   -0.000578 -0.002209 -0.000295 -0.000486 -0.000986  0.000754 -0.001423  0.000597
subject_holdout ridge              signed     row   0.15   -0.000457 -0.001045  0.003069  0.001614  0.001888 -0.004523 -0.002599 -0.001601
    interleaved ridge  risk_shrink_global  block6   0.65   -0.000454 -0.001331  0.000009 -0.000551 -0.001075  0.000503 -0.001625  0.000893
    interleaved ridge risk_shrink_subject  block6   0.40   -0.000392 -0.001430 -0.000195 -0.000317 -0.000670  0.000435 -0.000892  0.000324
    interleaved ridge  risk_shrink_global  block6   0.40   -0.000338 -0.000889 -0.000015 -0.000378 -0.000738  0.000243 -0.001100  0.000510
```

## Transfer And Placebo Gate

```text
          split model              family   level  param  mean_delta  placebo_p05  placebo_p50  placebo_p95  placebo_margin_vs_p05  pass_level_count                                                                                                                                                                 gate_verdicts
    interleaved ridge              signed subject   0.15   -0.001006    -0.001189    -0.000815     0.000215               0.000183                 7 Q1:PASS_LEVEL~submargin; Q2:PASS_LEVEL~submargin; Q3:PASS_LEVEL~submargin; S1:PASS_LEVEL~submargin; S2:PASS_LEVEL~submargin; S3:PASS_LEVEL~submargin; S4:PASS_LEVEL~submargin
    interleaved ridge              signed  block6   0.15   -0.000908    -0.001564    -0.000418     0.000986               0.000656                 7 Q1:PASS_LEVEL~submargin; Q2:PASS_LEVEL~submargin; Q3:PASS_LEVEL~submargin; S1:PASS_LEVEL~submargin; S2:PASS_LEVEL~submargin; S3:PASS_LEVEL~submargin; S4:PASS_LEVEL~submargin
    interleaved ridge              signed subject   0.30   -0.001861    -0.002224    -0.001475     0.000591               0.000363                 6           Q1:PASS_LEVEL~submargin; Q2:PASS_LEVEL~submargin; Q3:PASS_LEVEL~submargin; S1:DATE_BOUND; S2:PASS_LEVEL~submargin; S3:PASS_LEVEL~submargin; S4:PASS_LEVEL~submargin
    interleaved ridge              signed  block6   0.30   -0.001263    -0.002541    -0.000146     0.002724               0.001277                 6           Q1:DATE_BOUND; Q2:PASS_LEVEL~submargin; Q3:PASS_LEVEL~submargin; S1:PASS_LEVEL~submargin; S2:PASS_LEVEL~submargin; S3:PASS_LEVEL~submargin; S4:PASS_LEVEL~submargin
    interleaved ridge              signed  block6   0.50   -0.000811    -0.002754     0.001697     0.006986               0.001943                 6           Q1:DATE_BOUND; Q2:PASS_LEVEL~submargin; Q3:PASS_LEVEL~submargin; S1:PASS_LEVEL~submargin; S2:PASS_LEVEL~submargin; S3:PASS_LEVEL~submargin; S4:PASS_LEVEL~submargin
    interleaved ridge              signed subject   0.50   -0.002766    -0.003368    -0.002110     0.001354               0.000602                 5                     Q1:PASS_LEVEL~submargin; Q2:DATE_BOUND; Q3:PASS_LEVEL~submargin; S1:DATE_BOUND; S2:PASS_LEVEL~submargin; S3:PASS_LEVEL~submargin; S4:PASS_LEVEL~submargin
    interleaved ridge              signed subject   0.80   -0.003604    -0.004554    -0.002471     0.003099               0.000950                 4                               Q1:DATE_BOUND; Q2:DATE_BOUND; Q3:PASS_LEVEL~submargin; S1:DATE_BOUND; S2:PASS_LEVEL~submargin; S3:PASS_LEVEL~submargin; S4:PASS_LEVEL~submargin
subject_holdout ridge              signed     row   0.15   -0.000457     0.000172     0.001173     0.002632              -0.000629                 2                                                               Q1:LOTTERY; Q2:NO_GAIN; Q3:PASS_LEVEL~submargin; S1:NO_GAIN; S2:PASS_LEVEL~submargin; S3:LOTTERY; S4:DATE_BOUND
    interleaved ridge risk_shrink_subject  block6   0.65   -0.000578    -0.000349     0.000196     0.000667              -0.000229                 0                                                                                            Q1:LOTTERY; Q2:LOTTERY; Q3:LOTTERY; S1:LOTTERY; S2:NO_GAIN; S3:LOTTERY; S4:NO_GAIN
    interleaved ridge  risk_shrink_global  block6   0.65   -0.000454    -0.000381     0.000365     0.001061              -0.000073                 0                                                                                            Q1:LOTTERY; Q2:NO_GAIN; Q3:LOTTERY; S1:LOTTERY; S2:NO_GAIN; S3:LOTTERY; S4:NO_GAIN
    interleaved ridge risk_shrink_subject  block6   0.40   -0.000392    -0.000251     0.000083     0.000373              -0.000141                 0                                                                                            Q1:LOTTERY; Q2:LOTTERY; Q3:LOTTERY; S1:LOTTERY; S2:NO_GAIN; S3:LOTTERY; S4:NO_GAIN
    interleaved ridge  risk_shrink_global  block6   0.40   -0.000338    -0.000299     0.000156     0.000585              -0.000039                 0                                                                                            Q1:LOTTERY; Q2:NO_GAIN; Q3:LOTTERY; S1:LOTTERY; S2:NO_GAIN; S3:LOTTERY; S4:NO_GAIN
    interleaved ridge risk_shrink_subject  block6   0.20   -0.000210    -0.000140     0.000026     0.000171              -0.000070                 0                                                                                            Q1:LOTTERY; Q2:NO_GAIN; Q3:LOTTERY; S1:LOTTERY; S2:NO_GAIN; S3:LOTTERY; S4:NO_GAIN
    interleaved ridge  risk_shrink_global  block6   0.20   -0.000192    -0.000175     0.000051     0.000266              -0.000017                 0                                                                                            Q1:LOTTERY; Q2:NO_GAIN; Q3:LOTTERY; S1:LOTTERY; S2:NO_GAIN; S3:LOTTERY; S4:NO_GAIN
    interleaved ridge  risk_shrink_global subject   0.65   -0.000067    -0.000111    -0.000008     0.000127               0.000044                 0                                                                                            Q1:NO_GAIN; Q2:NO_GAIN; Q3:LOTTERY; S1:NO_GAIN; S2:LOTTERY; S3:NO_GAIN; S4:NO_GAIN
    interleaved ridge risk_shrink_subject subject   0.65   -0.000066    -0.000109    -0.000000     0.000107               0.000043                 0                                                                                            Q1:NO_GAIN; Q2:NO_GAIN; Q3:LOTTERY; S1:NO_GAIN; S2:LOTTERY; S3:NO_GAIN; S4:NO_GAIN
    interleaved ridge  risk_shrink_global subject   0.40   -0.000043    -0.000071    -0.000007     0.000076               0.000028                 0                                                                                            Q1:NO_GAIN; Q2:NO_GAIN; Q3:LOTTERY; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:NO_GAIN
    interleaved ridge risk_shrink_subject subject   0.40   -0.000042    -0.000069    -0.000002     0.000064               0.000026                 0                                                                                            Q1:NO_GAIN; Q2:NO_GAIN; Q3:LOTTERY; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:NO_GAIN
```

## Strongest Raw Feature Correlations

```text
target                                                          feature  corr_base_loss  corr_signed_residual
    Q1 subj_dev__ch2025_mUsageStats__sleep_00_09__app_total_time__delta          0.0536               -0.2025
    Q1           ch2025_mUsageStats__sleep_00_09__app_total_time__delta          0.0632               -0.2010
    Q1           ch2025_mUsageStats__sleep_06_09__app_total_time__delta         -0.0084               -0.1770
    Q1 subj_dev__ch2025_mUsageStats__sleep_06_09__app_total_time__delta         -0.0123               -0.1758
    Q1           ch2025_mUsageStats__sleep_09_12__app_total_time__delta          0.0210                0.1729
    Q1 subj_dev__ch2025_mUsageStats__sleep_09_12__app_total_time__delta          0.0278                0.1701
    Q1            ch2025_mUsageStats__sleep_00_09__app_total_time__last          0.0409               -0.1665
    Q1            ch2025_mUsageStats__sleep_06_09__app_total_time__last          0.0333               -0.1665
    Q1  subj_dev__ch2025_mUsageStats__sleep_00_09__app_total_time__last          0.0367               -0.1574
    Q1             ch2025_mUsageStats__sleep_09_12__app_max_time__delta          0.0013                0.1524
    Q1  subj_dev__ch2025_mUsageStats__sleep_06_09__app_total_time__last          0.0262               -0.1502
    Q1               ch2025_mUsageStats__sleep_03_06__app_max_time__q90          0.0183                0.1490
    Q2  subj_dev__ch2025_mUsageStats__sleep_03_06__app_max_time__median          0.1304               -0.0553
    Q2   subj_dev__ch2025_mUsageStats__sleep_03_06__app_total_time__std         -0.1278                0.0745
    Q2   subj_dev__ch2025_mUsageStats__sleep_03_06__app_total_time__q25          0.1243               -0.0584
    Q2                ch2025_mUsageStats__q_evening__app_mean_time__max         -0.0974                0.1219
    Q2           ch2025_mUsageStats__sleep_06_09__app_total_time__delta          0.0221               -0.1187
    Q2   subj_dev__ch2025_mUsageStats__sleep_03_06__app_max_time__first          0.1170               -0.0884
    Q2      subj_dev__ch2025_mUsageStats__q_evening__app_mean_time__max         -0.1137                0.1155
    Q2            ch2025_mUsageStats__sleep_06_09__app_total_time__last          0.0608               -0.1118
    Q2 subj_dev__ch2025_mUsageStats__sleep_06_09__app_total_time__delta          0.0125               -0.1095
    Q2             ch2025_mUsageStats__sleep_06_09__app_max_time__delta          0.0432               -0.1089
    Q2   subj_dev__ch2025_mUsageStats__sleep_00_12__app_total_time__max         -0.1086                0.0823
    Q2           ch2025_mUsageStats__sleep_00_09__app_total_time__delta          0.0737               -0.1077
    Q3             ch2025_mUsageStats__sleep_03_06__app_total_time__std          0.1642               -0.0949
    Q3             ch2025_mUsageStats__sleep_03_06__app_total_time__q90          0.1553               -0.0657
    Q3             ch2025_mUsageStats__sleep_03_06__app_total_time__q75          0.1504               -0.0606
    Q3           ch2025_mUsageStats__sleep_03_06__app_total_time__first          0.1483               -0.0828
    Q3             ch2025_mUsageStats__sleep_03_06__app_total_time__max          0.1396               -0.0562
    Q3   subj_dev__ch2025_mUsageStats__sleep_00_09__app_total_time__q90          0.1384               -0.1038
    Q3            ch2025_mUsageStats__sleep_03_06__app_total_time__mean          0.1382               -0.0585
    Q3             ch2025_mUsageStats__sleep_09_12__app_total_time__q10          0.0871               -0.1335
    Q3            ch2025_mUsageStats__sleep_00_09__app_mean_time__delta          0.1296               -0.0883
    Q3    subj_dev__ch2025_mUsageStats__sleep_03_06__app_mean_time__max         -0.0674                0.1279
    Q3             ch2025_mUsageStats__sleep_00_12__app_total_time__q25          0.0879               -0.1241
```

## Decision

- A sub-margin PASS_LEVEL signal exists, but its magnitude is far below the 0.54 requirement. It may be useful only as a tiny risk-controlled blend.
- Best raw OOF mean delta: `-0.003604`.
- Best gate-sorted mean delta: `-0.001006`.
- Any PASS_LEVEL target: `True`.

## Outputs

- `outputs/missingness_rhythm_error_probe_grid.csv`
- `outputs/missingness_rhythm_error_probe_gate.csv`
- `outputs/missingness_rhythm_error_probe_detection.csv`
- `outputs/missingness_rhythm_error_probe_future.csv`
- `outputs/missingness_rhythm_error_probe_feature_corr.csv`
- `outputs/missingness_rhythm_error_probe_summary.json`
