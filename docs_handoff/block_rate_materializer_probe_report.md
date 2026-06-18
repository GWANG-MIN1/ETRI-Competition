# Block-Rate Materializer Probe

E304 block residual predictions were translated into OOF row probabilities. No submission CSV was written.

## Best CV Rows

```text
                      view           split target_set            mode  amp  Q1_delta  Q2_delta      Q3_delta  S1_delta  S2_delta  S3_delta  S4_delta  mean_delta
fallback_calendar_basepred   block_random5     q_only          signed 0.15  0.000288 -0.000285 -6.959853e-04   0.00000  0.000000  0.000000  0.000000   -0.000099
fallback_calendar_basepred   block_random5     q_only          signed 0.08  0.000103 -0.000214 -4.240478e-04   0.00000  0.000000  0.000000  0.000000   -0.000076
fallback_calendar_basepred   block_random5     q_only          signed 0.25  0.000705 -0.000200 -9.245154e-04   0.00000  0.000000  0.000000  0.000000   -0.000060
fallback_calendar_basepred subject_holdout    s4_only         top_abs 0.08  0.000000  0.000000  1.110223e-16   0.00000  0.000000  0.000000 -0.000016   -0.000002
fallback_calendar_basepred   block_random5        all          signed 0.08  0.000103 -0.000214 -4.240478e-04   0.00036 -0.000362  0.000479  0.000108    0.000007
fallback_calendar_basepred subject_holdout    s4_only         top_abs 0.15  0.000000  0.000000  1.110223e-16   0.00000  0.000000  0.000000  0.000063    0.000009
fallback_calendar_basepred   block_random5    s4_only         top_abs 0.08  0.000000  0.000000  1.110223e-16   0.00000  0.000000  0.000000  0.000091    0.000013
fallback_calendar_basepred   block_random5    s4_only          signed 0.08  0.000000  0.000000  1.110223e-16   0.00000  0.000000  0.000000  0.000108    0.000015
fallback_calendar_basepred subject_holdout    s4_only          signed 0.08  0.000000  0.000000  1.110223e-16   0.00000  0.000000  0.000000  0.000116    0.000017
fallback_calendar_basepred subject_holdout     q_only          signed 0.08  0.000494  0.000767 -1.041513e-03   0.00000  0.000000  0.000000  0.000000    0.000031
fallback_calendar_basepred   block_random5    s4_only         top_abs 0.15  0.000000  0.000000  1.110223e-16   0.00000  0.000000  0.000000  0.000225    0.000032
fallback_calendar_basepred   block_random5    s4_only censor_conflict 0.08  0.000000  0.000000  1.110223e-16   0.00000  0.000000  0.000000  0.000259    0.000037
```

## Transfer/Placebo Gate

```text
                      view           split target_set            mode  amp  mean_delta  placebo_p05  placebo_p50  placebo_p95  placebo_margin  pass_level_count                                                                                                            gate_verdicts
fallback_calendar_basepred   block_random5        all          signed 0.08    0.000007    -0.000479    -0.000110     0.000311        0.000486                 2 Q1:DATE_BOUND; Q2:LOTTERY; Q3:DATE_BOUND; S1:PASS_LEVEL~submargin; S2:DATE_BOUND; S3:PASS_LEVEL~submargin; S4:DATE_BOUND
fallback_calendar_basepred   block_random5     q_only          signed 0.15   -0.000099    -0.000413    -0.000013     0.000607        0.000314                 0                                 Q1:DATE_BOUND; Q2:LOTTERY; Q3:DATE_BOUND; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:NO_GAIN
fallback_calendar_basepred   block_random5     q_only          signed 0.08   -0.000076    -0.000244    -0.000030     0.000300        0.000167                 0                                 Q1:DATE_BOUND; Q2:LOTTERY; Q3:DATE_BOUND; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:NO_GAIN
fallback_calendar_basepred   block_random5     q_only          signed 0.25   -0.000060    -0.000584     0.000081     0.001116        0.000524                 0                                 Q1:DATE_BOUND; Q2:LOTTERY; Q3:DATE_BOUND; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:NO_GAIN
fallback_calendar_basepred subject_holdout    s4_only         top_abs 0.08   -0.000002    -0.000104    -0.000001     0.000109        0.000102                 0                                       Q1:NO_GAIN; Q2:NO_GAIN; Q3:NO_GAIN; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:NO_GAIN
fallback_calendar_basepred subject_holdout    s4_only         top_abs 0.15    0.000009    -0.000181     0.000012     0.000220        0.000190                 0                                       Q1:NO_GAIN; Q2:NO_GAIN; Q3:NO_GAIN; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:NO_GAIN
fallback_calendar_basepred   block_random5    s4_only         top_abs 0.08    0.000013    -0.000096    -0.000017     0.000074        0.000109                 0                                    Q1:NO_GAIN; Q2:NO_GAIN; Q3:NO_GAIN; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:DATE_BOUND
fallback_calendar_basepred   block_random5    s4_only          signed 0.08    0.000015    -0.000094    -0.000012     0.000087        0.000110                 0                                    Q1:NO_GAIN; Q2:NO_GAIN; Q3:NO_GAIN; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:DATE_BOUND
fallback_calendar_basepred subject_holdout    s4_only          signed 0.08    0.000017    -0.000097     0.000011     0.000116        0.000114                 0                                       Q1:NO_GAIN; Q2:NO_GAIN; Q3:NO_GAIN; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:NO_GAIN
fallback_calendar_basepred subject_holdout     q_only          signed 0.08    0.000031    -0.000312     0.000089     0.000392        0.000343                 0                                       Q1:NO_GAIN; Q2:LOTTERY; Q3:LOTTERY; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:NO_GAIN
fallback_calendar_basepred   block_random5    s4_only         top_abs 0.15    0.000032    -0.000171    -0.000023     0.000146        0.000204                 0                                    Q1:NO_GAIN; Q2:NO_GAIN; Q3:NO_GAIN; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:DATE_BOUND
fallback_calendar_basepred   block_random5    s4_only censor_conflict 0.08    0.000037    -0.000014     0.000033     0.000079        0.000051                 0                                       Q1:NO_GAIN; Q2:NO_GAIN; Q3:NO_GAIN; S1:NO_GAIN; S2:NO_GAIN; S3:NO_GAIN; S4:NO_GAIN
```

## Decision

- The block-rate materializer does not clear the practical transfer bar; improvements are too small for a 0.54 path.
- Best OOF mean delta: `-0.000099`.
- Best gate-sorted placebo margin versus p05: `+0.000486`.
- Any PASS_LEVEL target: `True`.

## Outputs

- `outputs/block_rate_materializer_probe_grid.csv`
- `outputs/block_rate_materializer_probe_gate.csv`
- `outputs/block_rate_materializer_probe_summary.json`
- `outputs/block_rate_materializer_probe_report.md`
