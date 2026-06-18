# Current Anchor Candidate Audit

Old E35x/E36x row/action candidates were re-read as transferable per-subject level moves and compared with the current overshoot x0.8 anchor.

## Threshold

- FS LB: `0.5677475939`
- Current anchor LB: `0.5615333471`
- Target LB: `0.5400000000`
- Needed total gain vs FS: `-0.027748`
- Needed extra gain from current anchor: `-0.021533`

## Direct Transferable Level Scores

```text
                 label  public_lb_known  post_mean  worst_loose  p_improve    fav18  cos_current  orth_resid_l2  beats_054_mean  risk_nonpositive
current_overshoot_x0p8         0.561533  -0.005223    -0.001713   1.000000 1.000000     1.000000       0.000000           False              True
        e365_jackknife              NaN   0.033791     0.055657   0.000000 0.000000    -0.034606       5.511258           False             False
      e368_q2s1rowmask         0.576290   0.033791     0.055659   0.000000 0.000000    -0.034472       5.511159           False             False
       e363_cellrobust              NaN   0.033799     0.055672   0.000000 0.000000    -0.034546       5.512244           False             False
   e357_publicsurvival              NaN   0.033836     0.055851   0.000000 0.000000    -0.035051       5.512037           False             False
```

## Best Current-Anchor Mixes

```text
       kind               label  mix_alpha  post_mean  extra_vs_current_mean  worst_loose  p_improve  beats_054_mean  risk_nonpositive
   orth_add      e365_jackknife   0.100000  -0.003860               0.001363     0.001658   1.000000           False             False
   orth_add     e363_cellrobust   0.100000  -0.003860               0.001363     0.001658   1.000000           False             False
   orth_add    e368_q2s1rowmask   0.100000  -0.003860               0.001363     0.001659   1.000000           False             False
   orth_add e357_publicsurvival   0.100000  -0.003858               0.001365     0.001679   1.000000           False             False
replace_mix      e365_jackknife   0.100000  -0.003824               0.001399     0.001309   1.000000           False             False
replace_mix     e363_cellrobust   0.100000  -0.003824               0.001399     0.001309   1.000000           False             False
replace_mix    e368_q2s1rowmask   0.100000  -0.003823               0.001400     0.001310   1.000000           False             False
replace_mix e357_publicsurvival   0.100000  -0.003821               0.001401     0.001329   1.000000           False             False
   orth_add      e365_jackknife   0.250000  -0.001213               0.004010     0.007414   0.990550           False             False
   orth_add     e363_cellrobust   0.250000  -0.001212               0.004010     0.007414   0.991106           False             False
   orth_add    e368_q2s1rowmask   0.250000  -0.001212               0.004011     0.007415   0.989994           False             False
   orth_add e357_publicsurvival   0.250000  -0.001207               0.004016     0.007458   0.989994           False             False
replace_mix      e365_jackknife   0.250000  -0.000888               0.004335     0.006812   0.983324           False             False
replace_mix     e363_cellrobust   0.250000  -0.000887               0.004336     0.006811   0.983880           False             False
replace_mix    e368_q2s1rowmask   0.250000  -0.000886               0.004336     0.006813   0.983324           False             False
replace_mix e357_publicsurvival   0.250000  -0.000881               0.004342     0.006856   0.983324           False             False
   orth_add      e365_jackknife   0.500000   0.005172               0.010395     0.019079   0.000000           False             False
   orth_add     e363_cellrobust   0.500000   0.005175               0.010398     0.019083   0.000000           False             False
   orth_add    e368_q2s1rowmask   0.500000   0.005175               0.010398     0.019082   0.000000           False             False
   orth_add e357_publicsurvival   0.500000   0.005186               0.010409     0.019178   0.000000           False             False
```

## Decision

- No old row/action materializer has enough transferable level mass for a 0.54 route.
- No current-anchor mix reaches the 0.54 mean threshold.
- The best mixes only add tiny predicted mean gains over the current anchor, far below the `-0.021533` extra required.
- Known public evidence is also adverse: the row/action lineage with measured public LB remains around `0.576`, not near the current `0.5615` anchor.

## Outputs

- `outputs/current_anchor_candidate_audit_direct.csv`
- `outputs/current_anchor_candidate_audit_mix.csv`
- `outputs/current_anchor_candidate_audit_summary.json`
- `outputs/current_anchor_candidate_audit_report.md`
