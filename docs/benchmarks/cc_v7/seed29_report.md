# V7 seed 29: compact semantic/sensor training

Completed source-development screen, not fresh real-camera evidence. Five
matched 120-epoch arms, 1126 real SimpleCube++ TRAIN / 119 reused VAL rows.
The teacher uses only TRAIN views; no target-camera data or identity enters
estimator fitting. All five first-epoch CUDA replays are bitwise equal.
All ten best/final checkpoints passed independent CPU replay and FP64 rescoring.
Removing the training-only projection leaves predictions bitwise unchanged.

|Training arm|Best full mean °|Final full mean °|Combined risk80 °|
|---|---:|---:|---:|
|gt_native|2.4209|2.4852|1.9733|
|gt_sensor|2.6265|2.7542|2.1113|
|raw_teacher_sensor|2.6308|2.8676|2.1117|
|canonical_teacher_sensor|2.5506|2.7369|1.9836|
|canonical_teacher_native|2.4343|2.5790|1.9798|

Canonical-teacher minus raw-teacher C+ full mean: -0.0802°.
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
|100%|2.4209|2.6265|2.6308|2.5506|2.4343|
|95%|2.1856|2.4593|2.5101|2.4197|2.2898|
|90%|2.1203|2.2806|2.4061|2.3147|2.1762|
|80%|1.9733|2.1113|2.1117|1.9836|1.9798|
|70%|1.6523|1.9786|2.0209|1.6928|1.6260|
|60%|1.6764|1.4176|1.9340|1.4663|1.4025|

Virtual sensor diagnostic; these transformed source images are NOT captures
from real unseen cameras. Original FP64 GT is transformed by the same matrix.
The training/evaluation table above uses the stored FP32 GT promoted to FP64;
that declared precision difference is separately audited, not a model gain.

|Best checkpoint|Native °|Red gain °|Blue gain °|Small mix °|Training-range mix °|Extrapolation mix °|
|---|---:|---:|---:|---:|---:|---:|
|gt_native_best|2.4209|5.4943|16.0716|4.3990|12.9654|19.0137|
|gt_sensor_best|2.6265|4.2831|7.1070|2.8277|2.0958|1.8841|
|raw_teacher_sensor_best|2.6308|6.8207|7.7697|3.2692|2.4526|2.8431|
|canonical_teacher_sensor_best|2.5506|5.0268|8.1175|2.8472|2.2260|2.5220|
|canonical_teacher_native_best|2.4343|6.6277|14.1179|4.4795|13.4224|19.2164|
|historical_v2_direct|2.5901|5.2784|16.8397|4.3697|12.1131|17.9304|
|historical_v2_sog|3.2991|3.2991|3.2991|2.8421|2.2268|2.2350|

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
