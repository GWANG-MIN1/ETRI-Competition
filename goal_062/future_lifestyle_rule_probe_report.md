# Future Lifestyle Rule Probe

- Fit rows: `271`
- Tune rows: `79`
- Future rows: `100`
- Feature columns: `330`

## Future-Tail Results

```text
                        family      param  delta_Q1  delta_Q2  delta_Q3  delta_S1  delta_S2  delta_S3  delta_S4  mean_delta
        subject_prior_residual       -0.4 -0.007109 -0.004808 -0.027119  0.010379 -0.001457  0.006010 -0.002484   -0.003798
        subject_prior_residual       -0.2 -0.004311 -0.003387 -0.015061  0.004601 -0.001400  0.001846 -0.001709   -0.002775
        subject_prior_residual      -0.65 -0.008702 -0.003854 -0.038234  0.019589  0.000202  0.014247 -0.002260   -0.002716
     feature_risk_shrink_tuned per_target -0.005616  0.005739 -0.006175  0.000000  0.000000  0.002371  0.000000   -0.000526
        subject_prior_residual       -1.0 -0.007591  0.002677 -0.046370  0.037972  0.005598  0.031367  0.000137    0.003399
        subject_prior_residual        0.2  0.006094  0.005422  0.018389 -0.003547  0.002920  0.000787  0.002780    0.004692
        subject_prior_residual        0.4  0.014489  0.012993  0.040604 -0.006115  0.007662  0.004782  0.006873    0.011612
threshold_lifestyle_rule_tuned per_target  0.041118  0.031425  0.021203  0.036384  0.019746 -0.011654  0.014330    0.021793
        subject_prior_residual       0.65  0.030206  0.025745  0.074938 -0.008007  0.017053  0.016082  0.014631    0.024378
  subject_prior_residual_tuned per_target  0.006094  0.005422  0.140236  0.037972 -0.001457  0.060124 -0.001709    0.035240
    feature_signed_ridge_tuned per_target  0.300134  0.009541  0.058149 -0.000599  0.018423  0.004509  0.002019    0.056025
        subject_prior_residual        1.0  0.154792  0.050686  0.140236 -0.008205  0.045685  0.060124  0.041845    0.069309
```

## Tuned Rule Details

```text
                        family target  chosen_amp  tune_delta  lam    q                                                          feature direction     threshold  sign  alpha  tune_coverage  future_coverage
  subject_prior_residual_tuned     Q1         0.2   -0.000182  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
  subject_prior_residual_tuned     Q2         0.2   -0.000532  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
  subject_prior_residual_tuned     Q3         1.0   -0.034124  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
  subject_prior_residual_tuned     S1        -1.0   -0.019127  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
  subject_prior_residual_tuned     S2        -0.4   -0.007458  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
  subject_prior_residual_tuned     S3         1.0   -0.052306  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
  subject_prior_residual_tuned     S4        -0.2   -0.000697  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
    feature_signed_ridge_tuned     Q1         1.0   -0.029670  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
    feature_signed_ridge_tuned     Q2         0.2   -0.000992  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
    feature_signed_ridge_tuned     Q3         0.4   -0.018128  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
    feature_signed_ridge_tuned     S1        -0.2   -0.003472  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
    feature_signed_ridge_tuned     S2        -0.4   -0.018344  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
    feature_signed_ridge_tuned     S3         0.4   -0.015878  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
    feature_signed_ridge_tuned     S4         0.2   -0.002498  NaN  NaN                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
     feature_risk_shrink_tuned     Q1         NaN   -0.002233 0.85 0.85                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
     feature_risk_shrink_tuned     Q2         NaN   -0.002753 0.85 0.55                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
     feature_risk_shrink_tuned     Q3         NaN   -0.009056 0.85 0.55                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
     feature_risk_shrink_tuned     S1         NaN    0.000000 0.00 0.75                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
     feature_risk_shrink_tuned     S2         NaN    0.000000 0.00 0.75                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
     feature_risk_shrink_tuned     S3         NaN   -0.004695 0.85 0.85                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
     feature_risk_shrink_tuned     S4         NaN    0.000000 0.00 0.75                                                              NaN       NaN           NaN   NaN    NaN            NaN              NaN
threshold_lifestyle_rule_tuned     Q1         NaN   -0.042685  NaN 0.80 subj_dev__ch2025_mUsageStats__sleep_00_09__app_total_time__first       low 276052.400000  -1.0    0.5       0.594937             0.51
threshold_lifestyle_rule_tuned     Q2         NaN   -0.049559  NaN 0.50                ch2025_mUsageStats__q_evening__app_mean_time__max       low 298613.500000  -1.0    0.5       0.468354             0.34
threshold_lifestyle_rule_tuned     Q3         NaN   -0.027636  NaN 0.65             ch2025_mUsageStats__sleep_03_06__app_total_time__q10      high  31837.780000  -1.0    0.5       0.189873             0.23
threshold_lifestyle_rule_tuned     S1         NaN   -0.045878  NaN 0.35            ch2025_mUsageStats__sleep_00_12__app_mean_time__delta      high -16077.870238  -1.0    0.5       0.531646             0.47
threshold_lifestyle_rule_tuned     S2         NaN   -0.029571  NaN 0.35               ch2025_mUsageStats__q_evening__app_total_time__q25       low  87192.887500   1.0    0.5       0.455696             0.30
threshold_lifestyle_rule_tuned     S3         NaN   -0.021054  NaN 0.65   subj_dev__ch2025_mUsageStats__sleep_03_06__app_total_time__std       low  55101.913027  -1.0    0.5       0.151899             0.21
threshold_lifestyle_rule_tuned     S4         NaN   -0.029425  NaN 0.65                 ch2025_mUsageStats__q_21_09__app_max_time__first      high 298860.850000  -1.0    0.5       0.468354             0.41
```

## Decision

- There is a small forward-only glimmer, but the size is not enough for 0.54 and must be treated as a tiny blend at most.
- Best future mean delta: `-0.003798`.

## Outputs

- `outputs/future_lifestyle_rule_probe_grid.csv`
- `outputs/future_lifestyle_rule_probe_details.csv`
- `outputs/future_lifestyle_rule_probe_summary.json`
