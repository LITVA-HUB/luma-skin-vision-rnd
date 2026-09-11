# V7 seed 43: compact semantic/sensor training

Completed source-development screen, not fresh real-camera evidence. Five
matched 120-epoch arms, 1126 real SimpleCube++ TRAIN / 119 reused VAL rows.
The teacher uses only TRAIN views; no target-camera data or identity enters
estimator fitting. All five first-epoch CUDA replays are bitwise equal.
All ten best/final checkpoints passed independent CPU replay and FP64 rescoring.
Removing the training-only projection leaves predictions bitwise unchanged.

|Training arm|Best full mean °|Final full mean °|Combined risk80 °|
|---|---:|---:|---:|
|gt_native|2.4314|2.4842|1.9559|
|gt_sensor|2.5959|2.7619|2.0213|
|raw_teacher_sensor|2.5343|2.7330|1.8945|
|canonical_teacher_sensor|2.6528|2.6753|2.2310|
|canonical_teacher_native|2.4828|2.6820|2.0419|

Canonical-teacher minus raw-teacher C+ full mean: +0.1185°.
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
|100%|2.4314|2.5959|2.5343|2.6528|2.4828|
|95%|2.2656|2.4732|2.4220|2.5102|2.2910|
|90%|2.1435|2.3808|2.2035|2.5214|2.2004|
|80%|1.9559|2.0213|1.8945|2.2310|2.0419|
|70%|1.7120|1.8826|1.7615|1.7863|1.7572|
|60%|1.7135|1.6953|1.4171|1.6114|1.5502|

Virtual sensor diagnostic; these transformed source images are NOT captures
from real unseen cameras. Original FP64 GT is transformed by the same matrix.
The training/evaluation table above uses the stored FP32 GT promoted to FP64;
that declared precision difference is separately audited, not a model gain.

|Best checkpoint|Native °|Red gain °|Blue gain °|Small mix °|Training-range mix °|Extrapolation mix °|
|---|---:|---:|---:|---:|---:|---:|
|gt_native_best|2.4314|5.6427|16.1391|4.7027|14.0320|19.6148|
|gt_sensor_best|2.5959|5.3500|6.7490|3.0677|2.4499|2.4370|
|raw_teacher_sensor_best|2.5343|4.7420|7.3863|2.8208|2.0923|2.6064|
|canonical_teacher_sensor_best|2.6528|5.0221|7.8430|2.9057|3.2498|4.1347|
|canonical_teacher_native_best|2.4828|4.6934|15.1179|4.6719|16.1648|23.3470|
|historical_v2_direct|2.4391|5.3428|17.4759|4.7394|14.5483|20.8478|
|historical_v2_sog|3.3162|3.3162|3.3162|2.7843|2.0779|1.9385|

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
