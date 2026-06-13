"""Run all module level-estimators through the test-faithful + interleaved CV.
Validates the harness (recency helps Q2/Q3; rerank control doesn't; global Q-up is
CV-penalized) and produces the evidence table the synthesis reasons over."""
import hsjepa_core as C

m = C.load()
print("Frame:", m.shape, "| subjects:", m["subject_id"].nunique())
print("baseline test-faithful per-target logloss (held-out):")
base = C.cv_eval(C.est_baseline, proxy="test_faithful", n_seeds=12, frame=m)
print("  ", {t: round(base[t]["base_ll"], 4) for t in C.TARGETS})

print("\n" + "=" * 74)
print("SANITY (harness trust): recency should help Q2/Q3; rerank should NOT; global up penalized")
C.fmt(C.cv_eval(C.est_subject_recency, n_seeds=12, frame=m, tau=21.0, lam=1.0), "subject_recency tau=21 lam=1.0")
C.fmt(C.cv_eval(C.est_rerank_control, n_seeds=12, frame=m), "rerank_control (per-row, negative)")
C.fmt(C.cv_eval(C.est_global_shift, n_seeds=12, frame=m, delta=+0.08), "global +0.08 (Q-up probe; expect CV penalty)")

print("\n" + "=" * 74)
print("MODULE LEVEL-ESTIMATORS (test_faithful, 12 seeds)")
C.fmt(C.cv_eval(C.est_subject_recency, n_seeds=12, frame=m, tau=28.0, lam=1.0), "recency tau=28 (memory: robust)")
C.fmt(C.cv_eval(C.est_cohort_shrink, n_seeds=12, frame=m, shrink=0.3), "V131C cohort_shrink 0.3 (regression-to-mean)")
C.fmt(C.cv_eval(C.est_cohort_shrink, n_seeds=12, frame=m, shrink=-0.3), "cohort DE-shrink -0.3 (amplify extremes)")
C.fmt(C.cv_eval(C.est_transition_drift, n_seeds=12, frame=m, lam=0.5), "transition_drift lam=0.5 (State-Transition/Hysteresis)")
C.fmt(C.cv_eval(C.est_transition_drift, n_seeds=12, frame=m, lam=1.0), "transition_drift lam=1.0")

print("\n" + "=" * 74)
print("S-side global down ladder (S-down probe direction; CV view)")
for t_delta in (-0.02, -0.04):
    C.fmt(C.cv_eval(C.est_global_shift, n_seeds=12, frame=m, delta=t_delta), f"global {t_delta:+.2f} (all targets)")

print("\n" + "=" * 74)
print("CROSS-CHECK on interleaved proxy (the one that called the S234 failure)")
C.fmt(C.cv_eval(C.est_subject_recency, proxy="interleaved", n_seeds=8, frame=m, tau=28.0, lam=1.0), "recency tau=28 [interleaved]")
C.fmt(C.cv_eval(C.est_cohort_shrink, proxy="interleaved", n_seeds=8, frame=m, shrink=0.3), "cohort_shrink 0.3 [interleaved]")
C.fmt(C.cv_eval(C.est_rerank_control, proxy="interleaved", n_seeds=8, frame=m), "rerank_control [interleaved]")
