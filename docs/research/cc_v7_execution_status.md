# V7 execution and continuation state

CURRENT: all15 training arms and all30 best/final checkpoint replays complete.
Sessions23891,31047,82349,56677,38790 and91332 ended0. No V7 worker remains live.
All29 external methods were frozen in method_lock.json before target decoding;
SHA2562828472f40348ebb9785c8d294e4353fd6fc0d33956191756c074d8912e3948d.
External384/primary317 and sourceTEST462 preparation, all58 prediction files,
261 method/population metric records and the report completed. Independent
audit covers1,566 fixed-coverage/threshold cases, maximum discrepancy4.61e-11.

Primary canonical-teacher hypothesis failed: full5.5013 versus C+4.9761;
risk805.3135 versus4.7481. Strongest observed Fourier control4.3552 full and
3.6233 raw-risk80. Sensor-only augmentation helps versus native but is not
the strongest method. All three seeds and all controls remain reported.
[External evidence](../benchmarks/cc_v7_external/report.md).

The user has clarified direct skin color as the next primary endpoint.
An original CC BY4.0 regional RGB/instrument-XYZ pilot completed on100 sites
from20 held-out people. This is ordinary single-setup calibration, not V7.
Exact DeltaE white reference remains unverified. [Skin target](skin_color_target_2026_09_11.md).
Older process handles below are retained history, not instructions to restart.

Seed17: COMPLETE five120-epoch arms. All10 checkpoints independently replayed
on CPU; all five first-epoch full-state CUDA replays bitwise equal. Teacher
feature replay exactly matches at three fixed TRAIN positions for all four
views. The standard RISK/CAL pipeline, source selective curves and six fixed
virtual sensor cases completed. All162 scalar/ranking/threshold checks passed.
[Measured report](../benchmarks/cc_v7/seed17_report.md).

Live session23891: seed29 then seed43, sequential cc_v7_experiment.py, five
arms per seed. Do not launch another copy. Latest observed seed29 gt_sensor
after gt_native completed. Poll the existing session for authoritative state.
Seed17 session30030 ended0; verification/risk/stress session53841 ended0.
Teacher extraction43198 ended0. Risk-smoke53637 ended0. V6 session99152 ended0.

Before comparing real cameras, finish these same steps for seeds29/43:

1. cc_v7_verify.py --run experiments/runs/ccv7_semantic_sSEED --out
   docs/benchmarks/cc_v7/seedSEED_verification.json
2. cc_v7_risk.py --run experiments/runs/ccv7_semantic_sSEED --out
   experiments/runs/ccv7_semantic_risk_sSEED
3. cc_v7_source_risk.py --run experiments/runs/ccv7_semantic_sSEED --risk
   experiments/runs/ccv7_semantic_risk_sSEED --out docs/benchmarks/cc_v7/seedSEED_source_risk
4. cc_v7_stress.py --run experiments/runs/ccv7_semantic_sSEED --out
   docs/benchmarks/cc_v7/seedSEED_stress
5. cc_v7_audit_metrics.py --seed SEED; cc_v7_report.py --seed SEED.

Keep source input and all V7 core/training/teacher numerical bytes unchanged.
Training runs snapshot all numerical source; do not retrofit a newer source
identity onto old weights. Every output is immutable; existing outputs are
verified and reused, not overwritten. No external pixel/GT values were decoded
for V7. The next external population contains317 reference-history-disjoint
rows and384 sensitivity rows; method/weight/risk/preprocessing lock is still
required before decoding. The population alone is not a full method lock.

Preserved implementation failures, not scientific experiments:

- Initial teacher guard stopped before pixel/GT extraction because official
  TEST shares dates with TRAIN. Explicit metadata exception now permits that
  official split while rejecting fitting-role overlap. No test teacher targets.
- cc_v7_semantic_smoke_s17 and cc_v7_semantic_debug_s17 stopped at an inherited
  bounded-action assertion: FP32 normalize/log can round4 to4.0000004768.
  V7 now computes the same physical angle directly in RGB. Original failures
  and source snapshots remain. The two-epoch smoke2 passed all five arms and
  all10 independent checkpoint replays; its numbers are not accuracy evidence.
- Risk smoke initially stopped before output because the streaming reader needs
  globally sorted indices. A tested sorted-read/inverse-permutation preserves
  the RISK-then-CAL role ordering for both images and GT. All five fitted smoke
  pipelines subsequently completed. Never sort one array without its identities.
- The first verifier finished numerical checks but failed writing a missing
  output directory. Creating that directory and rerunning yielded final receipts;
  no training or prediction changed.

Full suite236 passed/14 historical ONNX warnings in36.41s; two subsequent
risk-role/alignment regression tests passed separately. Ruff passes new scripts.
This is active, unbounded R&D, not a completed or blocked goal.
