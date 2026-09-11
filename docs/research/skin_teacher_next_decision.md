# Decision after frozen pooled-teacher source readouts

Completed: 1,230 frozen licensed DINOv2 feature vectors and 108 Ridge fits on
original source skin photographs with real native instrument Lab. No teacher
fine-tuning, new image download, ordinary facial-phone test, or reserved
MSKCC TEST/CAL / UMINHO held-out access. Source validation has been repeatedly
examined. Original independent skin accuracy remains unchanged.

## Result and strongest opponent

| Selected source readout | Mixed mean DeltaE00 | SLR to iPod | iPod to SLR |
|---|---:|---:|---:|
| Color36 | 4.5139 | 7.8220 | 8.2597 |
| Teacher768 | 4.2651 | 6.3118 | 7.5572 |
| Color + teacher | 3.9462 | 5.6611 | 8.2684 |
| Color + shuffled teacher, seed17 | 4.7328 | 5.8550 | 6.1725 |
| Color + shuffled teacher, seed29 | 4.7756 | 6.0047 | 5.8540 |
| Color + shuffled teacher, seed43 | 4.6024 | 6.0191 | 6.0944 |
| Strong compact capture mixture, mean of 3 seeds | 3.4406 | 5.0193 | 5.9635 |
| Strong compact capture plain, mean of 3 seeds | 3.4771 | 5.8301 | 5.5089 |
| Training-only graph, mean of 3 seeds | 3.6394 | 5.8339 | 4.9736 |

Teacher features add information for the pooled linear readout on mixed and
SLR-to-iPod source data. Shuffling removes most of that gain. However, the
combined readout loses to a strong compact model in every protocol, and is
worse than all shuffled controls in the reverse direction. Those controls
select different regularization strengths on same-camera validation; their
performance is not proof of a beneficial random-feature mechanism or pure
camera causality. Never choose alpha using the other-camera errors.

Combined error at 80% coverage is 3.8573 / 6.0811 / 8.3628. Density-based
rejection worsens both transfer means. It is not expected-error calibration.
There is no model promotion, selective-risk victory or universality claim.

## What this does and does not test

The teacher is frozen and its CLS plus mean patch token is used. Each readout
is linear after fitting-only column/block normalization. This tests accessible
information under that representation/readout family, not the best attainable
result from all DINOv2 fine-tuning, local-token or nonlinear architectures.
Color36 and teacher768 have different feature capacity; combined and shuffled
controls have the same 804-feature readout. All receive six alpha candidates.
Tiny readout parameter counts do not make this a tiny image model: the teacher
has 22,056,576 parameters and 88,283,115 weight bytes. Combined readout adds
2,415 coefficients/intercepts. It exceeds the intended inference budget.

The input is existing 128px source RGB upsampled to 224, not new high-resolution
detail. Camera cohorts contain different people; source transfer mixes sensor,
capture and population shifts. Teacher pretraining overlap remains unknown.

All 1,230 feature vectors exactly replay at the same device/batching. All 216
readout prediction arrays replay; 38,016 independent scalar DeltaE00 cases,
648 fixed-coverage rows, and 108 weighted normal-equation/direct-solve checks
pass. Every one of the 108 candidate fits and 19,008 risk-curve points remains.

## Next bounded hypothesis: local correspondence instead of global pooling

Do not launch a major distillation sweep for this pooled representation. The
assumption to attack next is that global semantic averaging retains the useful
skin information. It removes the explicit association between a local surface
descriptor and the RGB values measured at that location.

Mechanism hypothesis: spatially aligned frozen teacher patch descriptors may
help a predictor weight surface regions (texture/artifacts/capture effects)
while preserving the corresponding absolute RGB. Keep local RGB features and
test aligned descriptors against global descriptors broadcast everywhere and
descriptors shuffled across locations within the same image. This separates
spatial correspondence from teacher capacity and extra parameters.

Key assumption: local teacher features describe something useful about the
visible tissue/capture that our hand-engineered patch summaries omit. Failure
mode: teacher tokens describe camera artifacts or homogeneous texture without
reference-color information. Cheapest falsifier: one bounded frozen-token
readout screen with identical heads and matched ordinary controls, in both
transfer directions. This remains PLANNED, not measured in this cycle.
Do not distill or deploy unless the teacher-assisted readout first provides
a convincing advantage over the strongest ordinary compact comparator.

Paired common-bias reweighting, source spectral oracles and synthetic negatives
remain intact. Primary independent MSKCC result stays 4.4570 full / 4.1591 at
80%; ordinary fusion 4.3005 / 4.1447 remains stronger. The ordinary unseen-phone
facial target median<=2 / p95<=5 at>=80% coverage remains an aspiration.

[Full measured screen](../benchmarks/skin_teacher_readout_v1/report.md),
[frozen protocol](skin_teacher_readout_protocol_v1.md),
[original teacher rights](../ip/dinov2_teacher_adoption.md).
