# Actual skin-color support and representation evidence

Implemented: nested person-disjoint internal roles, equal-update learning curves,
original compact baseline, capacity-matched statistical/pixel residual adapters,
and frozen-model local-information interventions. Camera identity is not an input.

Measured: 27 fits on real original MSKCC skin photographs and native instrument
Lab. Increasing training people6 to18 improves baseline mean7.2439 to6.1082 on
the fixed six-person internal cohort; learned pixels do not outperform the
baseline. Texture removal scarcely changes pixel-model outputs. These are
internal exploratory results, not fresh independent or smartphone evidence.
[Decision and report](../research/skin_support_curve_next_decision.md).

Reproducibility:27 exact prediction arrays,3 complete state refits,6264 scalar
color cases,162 coverage rows; diagnostics add21426 scalar color cases and
267264 patch intervention checks.353 tests pass,14 historical warnings.
Model size0.929-0.941M parameters; maximum measured training allocation318.36MiB.
No new isolated inference latency/export claim or novelty claim.

Still unvalidated: ordinary phone facial color precision, robust unseen-camera
behavior, reliable refusal, commercial cosmetics matching and technical advantage
over the strongest independent-test baseline. No legal classification implied.
