# Pooled visual-teacher source evidence for native skin color

Implemented: original licensed frozen DINOv2 extraction and weighted linear
readouts using real MSKCC instrument Lab. 108 source readout fits cover color,
teacher, combined and shuffled controls, with explicit camera-source protocols.
All use one image. Source validation is reused research data, not a fresh test.

Measured: combined pooled features improve the weak linear color readout in
two protocols but fail to beat strong compact nonlinear models in every
protocol. Reverse transfer is worse than shuffled controls. Density rejection
does not yield a selective advantage. No novel or deployable model is claimed.

The teacher has 22.06M parameters, beyond the target local model budget. Its
tiny readout does not remove the teacher inference requirement. A compact
student has not been trained from this representation. Original Apache 2.0
notices and weight/source hashes were verified; no additional dataset or
weights were acquired. Upstream pretraining overlap is unknown.

Verified: 1,230 exact feature vectors, 216 exact readout arrays, 38,016 scalar
DeltaE00 checks, 648 coverage checks and 108 independent linear-system checks.
The previous independent test and ordinary facial-phone validation status are
unchanged. No participant images/IDs or model weights are committed.

[Results and next falsifiable hypothesis](../research/skin_teacher_next_decision.md).
