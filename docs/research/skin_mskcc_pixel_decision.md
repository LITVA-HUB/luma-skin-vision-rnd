# Decision after direct skin pixel screens (2026-09-11)

## What changed

Actual local JPEG -> skin crop -> features/network -> instrument Lab now works.
This goes beyond the earlier author-feature pilot. All three compact patch-vote
seeds outperform the standard MobileNetV3-small and approximately capacity-
matched global-color MLP on the SAME six-person source-development set.

|Single model, source mean DeltaE00|Seed17|Seed29|Seed43|Parameters|
|---|---:|---:|---:|---:|
|MobileNetV3-small|3.8139|4.0029|4.1147|1520931|
|Global-color MLP|3.8113|3.7303|3.9020|938755|
|Confidence-weighted patch votes|3.4888|3.5074|3.4964|924932|

This supports keeping patch/color-distribution evidence as a useful compact
representation. It does not prove a new architecture category, novelty, camera
independence or efficacy on unseen people outside the development partition.
All24TRAIN people,6VALIDATION people and source modality/camera distributions
remain the same. No test or calibration participant was numerically opened.

## Mechanisms that did NOT produce the hoped-for advance

- Three Huber5updates barely changed the result; only0.01–0.70% of votes at
  selected checkpoints exceeded that residual threshold.
- Tenfold tighter Huber0.5 did not consistently improve single-seed models.
  Its three-seed ensemble has mean3.4275 versus3.4588 for plain pooling, but
  this small adaptive-source improvement is not a reliable mechanism victory.
- Forcing local color votes to ignore global context worsened the result.
  The context is useful; making hypotheses more independent is not sufficient.
- Histogram MLPs lost simpler features and hit the fixed2000iteration limit.
- A fixed50/50CNN+patch blend helps seed17 but not all seeds. A six-model blend
  reaches source mean3.3906 / median2.9094 / p956.6950, but uses7337589parameters.
  Its descriptive participant-bootstrap difference versus the three-patch
  ensemble includes zero. Do not confuse that ensemble with the924932parameter
  single model or claim its accuracy at single-model latency.

## Reliability and compute

For the three-patch ensemble, meanDeltaE00 at80% density-ranked coverage is
3.2426 versus3.3885 for the three-CNN ensemble, using the same density score.
Patch seed disagreement alone gives3.4491 at80%, almost no reduction from full
3.4588. This is diagnostic ranking, not calibrated predicted DeltaE00.
Complete source curves are archived in1056CSV rows and the plotted60–100% range.

RTX4060 seed17 patch model: model-plus-Lab-decode median GPU0.4915ms; parameter
count924932; checkpoint3704630bytes; training peak108.407MiB including descriptor
caches; inference allocated peak13.726MiB. These exclude JPEG preparation.
The current CPU path on10TRAIN originals takes median91.79ms / p95134.10ms,
including hash/file read, decode/crop/resize and ALL feature extraction. No
skin/face localization is included. MobileNet model median3.1438ms separately.
No ONNX/TensorRT optimization or deployment validation was performed.

## Next decision

Do not expand the architecture search indefinitely against these six people.
Prepare a matched downstream-error experiment: subject-out-of-fold residuals
on24TRAIN people, standard compact error head versus disagreement-aware risk,
then post-hoc calibration on the reserved6people. Freeze method/checkpoint/risk
definitions before exposing the10-person final test. Test must include the
strongest locally reproduced baseline and all retained proposed comparators.

No camera-independent skin-color result exists yet. Both available MSKCC
devices have already appeared in development. A camera-excluded refit would
be an exploratory transfer comparison; it cannot undo target-camera exposure
in architecture selection or prove generalization to an entirely new phone.
Strict new-camera skin accuracy needs additional legitimate instrument-paired
data or a separately justified protocol. Do not substitute angular benchmarks.

Evidence: [v1](../benchmarks/skin_mskcc_pixels_v1/report.md),
[v2](../benchmarks/skin_mskcc_pixel_ablation_v2/report.md), source-only fusion,
profile and scalar/replay audits in the same benchmark folders. All42best/final
checkpoints replay exactly.256tests pass;14historical ONNX warnings remain.
