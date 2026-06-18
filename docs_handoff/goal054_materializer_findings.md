# Goal 0.54 Materializer Findings

Date: 2026-06-19

## Starting Point

- Current best public LB: `0.5615333471`
- Target LB: `0.5400000000`
- Extra improvement required from current best: `-0.021533`

## Experiment 1: Block-Rate Materializer Probe

The block residual signal was translated into OOF row probabilities and vetted with transfer/placebo gates.

- Grid size: `150`
- Best OOF mean delta: `-0.000099`
- Best gate-sorted row: `+0.000007` mean delta
- PASS_LEVEL markers: only `PASS_LEVEL~submargin`, not floor-clearing
- Practical decision: signal exists, but it is roughly two orders of magnitude too small for a 0.54 route.

Outputs:

- `outputs/block_rate_materializer_probe_report.md`
- `outputs/block_rate_materializer_probe_grid.csv`
- `outputs/block_rate_materializer_probe_gate.csv`
- `outputs/block_rate_materializer_probe_summary.json`

## Experiment 2: Current-Anchor Re-Audit of E35x/E36x Row/Action Candidates

Old row/action candidates were re-read as transferable per-subject level moves, then compared with the current overshoot x0.8 anchor.

- Needed total gain vs FS: `-0.027748`
- Current anchor post-mean gain vs FS: `-0.005223`
- Best direct candidate: current anchor itself
- Old E35x/E36x direct level moves: about `+0.0338` predicted loss, not gain
- Best mix with current anchor: `-0.003860`, which is `+0.001363` worse than current
- Any mix reaches 0.54 mean threshold: `false`
- Any safe mix reaches 0.54: `false`

Outputs:

- `outputs/current_anchor_candidate_audit_report.md`
- `outputs/current_anchor_candidate_audit_direct.csv`
- `outputs/current_anchor_candidate_audit_mix.csv`
- `outputs/current_anchor_candidate_audit_summary.json`

## Extra Check: H001/H002 Q2/S1 E247-Locked Line

H001 and H002 were inspected because they looked like post-E368 Q2/S1 action translators. Both reports are diagnostic-only:

- H001 selected no submission-ready file.
- H002 selected no submission-ready file.
- Both weaken the idea that row selection alone can safely translate the Q2/S1 latent action.

## Conclusion

No honest 0.54 route was found. The only observed path that can numerically cross 0.54 remains public-equation/cell optimization, but previous stress checks showed it requires large hidden-risk exposure. The safe current choice remains the `0.5615333471` overshoot x0.8 anchor.
