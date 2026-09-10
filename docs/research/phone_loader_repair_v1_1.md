# Phone evaluator loader repair v1.1

Original v1 lock and executable remain untouched. Preparation completed88 inputs.
Prediction failed on the FIRST V2 checkpoint before producing any prediction
file or model error. The old V2 artifact is a mapping with keys `state`, `epoch`;
the new runner incorrectly passed that outer mapping to load_state_dict.
The empty v1 predictions directory is preserved. V5 checkpoints are raw state
mappings and need no unwrap. Model weights, protocols, data, reference policy,
calibration and method choices are unchanged. No phone model errors were seen.

New executable cc_phone_benchmark_v1_1.py changes ONLY checkpoint unwrapping and
its output directory to predictions_v1_1. Its digest is bound in a separate
repair receipt committed before first predictions/errors. Original checks still
verify every frozen artifact. The prediction/evaluation bodies otherwise remain
identical; preflight runs on two NT samples from the TRAIN loader pool verify all30 signatures.
This corrects a file-format bug, not a scientific post-test protocol revision.
