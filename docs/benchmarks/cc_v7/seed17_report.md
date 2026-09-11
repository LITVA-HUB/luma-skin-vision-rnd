# V7 seed 17: compact semantic/sensor training

Completed source-development screen, not fresh real-camera evidence. Five
matched 120-epoch arms, 1126 real SimpleCube++ TRAIN / 119 reused VAL rows.
The teacher uses only TRAIN views; no target-camera data or identity enters
estimator fitting. All five first-epoch CUDA replays are bitwise equal.
All ten best/final checkpoints passed independent CPU replay and FP64 rescoring.
Removing the training-only projection leaves predictions bitwise unchanged.

|Training arm|Best full mean °|Final full mean °|Combined risk80 °|
|---|---:|---:|---:|
|gt_native|2.3383|2.5309|1.7595|
|gt_sensor|2.4592|2.5583|2.0054|
|raw_teacher_sensor|2.4349|2.6404|1.7608|
|canonical_teacher_sensor|2.4591|2.5585|1.8630|
|canonical_teacher_native|2.5890|2.6565|2.1840|

Canonical-teacher minus raw-teacher C+ full mean: +0.0243°.
A negative difference favors the candidate. This is a single training seed on
reused validation; neither direction establishes novelty or external transfer.
A target corrected by known training illumination is an approximate camera-RGB
view, not a measured intrinsic surface color or facial Lab reference.

Standard risk heads use 259 held-out RISK rows with five capture-group folds;
268 date-disjoint CAL rows provide a positive scale and fixed thresholds.
Each arm has the same five candidate heads for context, cheap and combined
features. Combined is predeclared primary; the others are archived ablations.
Below, nominal 80% coverage accepts 95/119 = 79.83% per model. All curves and
source-calibrated threshold coverages are in the machine-readable results.

|Coverage|GT native|GT sensor|Raw teacher + sensor C+|Canonical teacher + sensor|Canonical teacher native|
|---|---:|---:|---:|---:|---:|
|100%|2.3383|2.4592|2.4349|2.4591|2.5890|
|95%|2.0553|2.1759|2.3158|2.2034|2.3723|
|90%|1.9245|2.1408|2.1139|2.0048|2.2818|
|80%|1.7595|2.0054|1.7608|1.8630|2.1840|
|70%|1.4516|1.6020|1.5787|1.6762|1.6581|
|60%|1.2259|1.4868|1.4460|1.3455|1.4894|

Virtual sensor diagnostic; these transformed source images are NOT captures
from real unseen cameras. Original FP64 GT is transformed by the same matrix.
The training/evaluation table above uses the stored FP32 GT promoted to FP64;
that declared precision difference is separately audited, not a model gain.

|Best checkpoint|Native °|Red gain °|Blue gain °|Small mix °|Training-range mix °|Extrapolation mix °|
|---|---:|---:|---:|---:|---:|---:|
|gt_native_best|2.3383|6.4074|14.7779|4.4170|13.1345|19.4090|
|gt_sensor_best|2.4592|4.3588|6.2588|2.9024|2.1371|2.6372|
|raw_teacher_sensor_best|2.4349|4.7896|7.9668|2.8593|2.4705|2.6389|
|canonical_teacher_sensor_best|2.4591|5.4525|9.2924|3.0760|3.2709|4.0406|
|canonical_teacher_native_best|2.5890|5.5042|16.4710|4.6579|14.3561|21.6787|
|historical_v2_direct|2.4138|5.6597|14.9731|4.7196|14.3667|21.0325|
|historical_v2_sog|3.2450|3.2450|3.2450|2.6588|1.9047|1.5651|

The archived diagnostics also report prediction equivariance errors, p95 and
catastrophic tails. The preserved SoG baseline must pass the diagonal-gain
consistency check. Zero equivariance error does not imply correct illumination.
All matrix cases and best/final results were retained, including regressions.

Model: 3,033,651 deployment parameters plus 369,024 training-only projection
parameters. The saved training checkpoint is 13,816,203 bytes; it includes the
projection. Seed17 training peak is 932.71–933.83 MiB including 233.45 MiB source
cache, 116.72 MiB teacher caches and the initial state. Other seeds have their
own allocation records. V7 deployment file size, latency and ONNX/TensorRT
results are NOT MEASURED. The teacher itself is absent at inference.

Source data: SimpleCube++, original CC BY4.0. Teacher: standard original
DINOv2-S/14, Apache2.0; pretrained-image overlap cannot be independently audited.
Its train-only feature cache took 238.88 seconds on CPU. Three fixed train
images across four view types replayed exactly; all cache rows have hash/ID
bindings. Images/teacher features/weights are excluded from this Git archive.

Next decision remains the prespecified three-seed comparison and frozen
317-row real INTEL-TAU transfer population (384-row sensitivity). No new target
pixels or GT values have been decoded for this report. No arbitrary-camera,
ordinary phone JPEG/HEIC, facial skin accuracy or finite-sample risk guarantee.
