# V7 component evidence and limits

Candidate positioning remains: compact adaptive color-normalization and
reliability estimation technology for camera-independent analysis of
color-sensitive visual objects, with first intended application in facial
skin analysis and cosmetics recommendation. This is proposed technical
positioning, not an approved legal classification or a novelty determination.

Implemented: one-image compact estimation; ordinary matched error heads and
held-out source calibration; training-only semantic supervision; physically
limited sensor-response augmentation; deterministic training replay; independent
metric/checkpoint auditing. V7's deployment estimator has3,033,651 parameters.
The22M teacher and0.369M projection are training-only and removed at inference.
No cloud or target-camera identity/calibration is required by these estimators.

Measured now on real public source GT: first V7 seed, five120-epoch arms on
SimpleCube++1126 TRAIN/119 reused VAL. Canonical-teacher+sensor full mean2.4591°
and risk80 1.8630° versus matched raw-teacher C+2.4349° and1.7608°. The proposed
training target does NOT improve these primary source metrics. All10 saved
best/final checkpoints passed CPU replay and independent error rescoring.
The additional two seeds are running. All failures are preserved.

Virtual sensor diagnostic: GT-native19.4090° versus GT-sensor2.6372° on the
fixed extrapolation matrix. Historical SoG is stronger there,1.5651°. This is
a controlled transformation of real source images/labels, not a real-camera
measurement or proof of universal sensor generalization. No new V7 external
camera result, deployment latency or ONNX/TensorRT export is reported.

Existing real external evidence remains the separately frozen V2 INTEL-TAU
and Samsung/Oppo transfer screens. [Phone report](../benchmarks/phone_v1_alias_report.md)
records V2SoG4.367° full /3.951° risk80 versus direct C+4.838° /4.498° on79
scorable references; paired confidence intervals include zero and behavior
differs by camera. V5 did not improve that phone result. V6's three-seed source
combination screen is also negative. None of these outcomes is hidden by V7.

Data/weight lineage: SimpleCube++ original CC BY4.0; standard DINOv2 original
Apache2.0 teacher, with unverified pretraining/benchmark overlap disclosed.
New INTEL-TAU population is metadata-only frozen,317 primary plus384 sensitivity
rows under original CC BY-SA4.0. In V7 it is reserved for evaluation, not
estimator or risk training. The sparse-mirror provenance limit remains explicit.
No dataset images or weights have been published externally.

Still unvalidated for Luma: facial skin colorimetry, corresponding surface
CIEDE2000, arbitrary smartphone JPEG/HEIC processing, all-camera robustness,
calibration under unrestricted distribution shift, and any patentable novelty.
Appropriate facial reference measurements remain future work. Public illuminant
GT validates aspects of the normalization/reliability component; it does not
measure the accuracy of final skin-color or cosmetics recommendations.

[V7 source report](../benchmarks/cc_v7/seed17_report.md),
[V6 negative report](../benchmarks/cc_v6_report.md),
[next transfer protocol](../research/cc_v7_transfer_diagnostics_protocol.md),
[active execution](../research/cc_v7_execution_status.md).
