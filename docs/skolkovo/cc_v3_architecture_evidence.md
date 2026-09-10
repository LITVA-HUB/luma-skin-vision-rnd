# V3 architecture exploration evidence —2026-09-10

Candidate positioning remains: compact adaptive color-normalization and reliability estimation for camera-independent analysis of color-sensitive visual objects, with facial skin analysis and cosmetics recommendation as the first intended application. This is technical positioning, not an approved legal classification.

**Implemented:** an image-dependent color-frame graph with a directional posterior and approximate camera-space reproduction-risk transport; matched direct/diagonal graph controls; explicit conditioning/support diagnostics; an original-source-grounded FFCC numerical control; immutable data/experiment provenance.

**Measured on real public data:** three120epoch source training runs on1,126 SimpleCube++ images, checkpoint selection and analysis on119 previously used development-validation images. Full-frame Proposed loses to the matched direct graph,4.281° versus2.547° mean reproduction; raw selective risk80 also loses,4.155° versus2.304°. This is negative R&D evidence and cannot support a claim of a newly superior component. The priorV2 independent-image camera-transfer result is separate and remains preserved.

**Engineering feasibility measured:**1.216M parameters and643.09MiB peak allocated training memory including the source cache on RTX4060. New architecture inference latency/export are unmeasured. Newly acquired multicamera images are not yet evaluated or used for training.

**Still to validate:** a revised architecture advantage over strong matched baselines; calibrated selective risk on locked unseen cameras; robustness under smartphone processing; facial surface-color reference measurements; real skin Lab/ΔE00 accuracy; cosmetics utility. Generic illuminant ground truth cannot fill the skin-specific validation gap.

**IP interpretation:** the full-frame canonicalization pattern has published antecedents, including GL frame methods. No novelty, patentability or freedom-to-operate conclusion is supported. The contribution hypothesis must be narrowed and tested. The failing design, source licenses and source-level implementation evidence are retained as research records.

See [V3 report](../benchmarks/cc_v3_report.md), [decision](../research/cc_v3_revision_decision.md), [prior art](../research/cc_v3_prior_art.md), [data terms](../data/cc_v3_usage_decision.md) and [V2 evidence](cc_v2_evidence_addendum.md).
