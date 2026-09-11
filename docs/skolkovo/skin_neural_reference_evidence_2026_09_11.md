# Larger compact skin-color adapter: implemented, measured, not a novelty win

IMPLEMENTED: three matched193,795-parameter correction heads attached to the
same929,297-parameter frozen neural core. Total1,123,092 parameters within the
user-authorized cap. One image supplies learned context and color descriptors;
local-reference variants use fixed TRAIN memory with no inference camera ID.
Training queries exclude all reference images of their person from that memory.

MEASURED ON PUBLIC REAL DATA:27 fits on MSKCC CC-BY instrument-native skin Lab,
with three seeds and explicit source camera-held-out protocols. Local affine
improves its core forward5.0193 ->4.9658 DeltaE00, but worsens mixed3.4406 ->
3.5182 and reverse5.9635 ->6.4782. It does not consistently beat the ordinary
matched residual head, and it loses stronger historical transfer comparators.
All27 heads lower training MSE; that does not establish generalization.

RESOURCE EVIDENCE: affine prepared-feature batch1 median1.424–3.042ms on
RTX4060; complete checkpoint4.53–4.58MB including reference memory. This excludes
JPEG decode/feature extraction. Allocated training memory is measured with
cached features; no isolated deployment VRAM or ONNX/TensorRT export claim.

STILL TO VALIDATE: ordinary facial skin on uncalibrated smartphones, meaningful
refusal calibration, cosmetics decisions and superiority of a technically
distinct mechanism over matched standard models. Existing independent skin
test remains primary4.4570/80%4.1591 versus stronger ordinary fusion4.3005/
4.1447. No new independent test was opened; current source validation is reused.

Candidate positioning remains compact adaptive color-normalization and
reliability technology for color-sensitive visual analysis, with a first
commercial application in skin analysis. This is proposed positioning, not an
approved legal classification. Known differentiable/local regression mechanisms
must not be described as an invention merely because this implementation is new.

[Audited report](../benchmarks/skin_neural_reference_v1/report.md),
[next research decision](../research/skin_neural_reference_next_decision.md).
