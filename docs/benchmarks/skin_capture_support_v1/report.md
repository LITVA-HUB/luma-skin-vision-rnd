# Observed same-site patch support experiment

45locally reproduced fits. Evaluation inputs are original source photographs;
targets are actual MSKCC instrument-native Lab. All errors are skin DeltaE00.
This is heavily reused source development, not independent TEST or phone validation.
Prefit checkpoint: `checkpoint/skin-capture-support-pretrain-2026-09-11`.

TRAIN has1,421same-site pairs with exactly equal native references, but zero
cross-camera pairs. Spatial pattern similarity does not prove registration.
Paired augmentation mixes exact observed patch tokens, without RGB interpolation.
Mixed bags are derived training inputs, not new real photographs or measurements.

## Actual source color errors

Cells average three separate seed scores, not an ensemble. p95 is the mean
of individual seed p95 values. All arms have identical inference architecture.

| Protocol | Arm | Mean | Median | p95 | At80% |
|---|---|---:|---:|---:|---:|
| mixed | baseline | 3.4406 | 2.9776 | 6.9316 | 3.4407 |
| mixed | self_bootstrap | 3.4509 | 2.9476 | 7.2967 | 3.3923 |
| mixed | soft_mode_control | 3.4654 | 3.0605 | 7.2623 | 3.5426 |
| mixed | paired_union | 3.5614 | 3.0369 | 7.2256 | 3.5978 |
| mixed | paired_stratified | 3.5619 | 3.0658 | 7.2012 | 3.6018 |
| from_SLR | baseline | 5.0193 | 4.8340 | 9.7825 | 5.1087 |
| from_SLR | self_bootstrap | 5.0214 | 4.8120 | 9.8668 | 5.0660 |
| from_SLR | soft_mode_control | 4.9546 | 4.8344 | 9.5594 | 4.9891 |
| from_SLR | paired_union | 4.8620 | 4.6519 | 9.1931 | 4.8488 |
| from_SLR | paired_stratified | 4.8328 | 4.7462 | 9.2002 | 4.8525 |
| from_ipod | baseline | 5.9635 | 5.5527 | 11.0896 | 5.7914 |
| from_ipod | self_bootstrap | 5.8964 | 5.5129 | 10.9336 | 5.7661 |
| from_ipod | soft_mode_control | 6.1421 | 5.8015 | 10.9571 | 6.0659 |
| from_ipod | paired_union | 5.9336 | 5.6443 | 10.6749 | 5.7886 |
| from_ipod | paired_stratified | 5.9520 | 5.6234 | 10.7105 | 5.8597 |

self_bootstrap controls within-image resampling. soft_mode_control changes only
the auxiliary mode target, so paired-versus-soft comparisons isolate input mixing.
paired_union samples from both bags with replacement; paired_stratified preserves
one observed token at each grid index, without assuming physical pixel alignment.

## Matched contrasts

Patient resampling intervals are descriptive on reused source data, not
confirmatory or multiplicity-adjusted. Camera protocols also change people
and capture composition. Negative difference favors the candidate.

| Protocol | Candidate vs control | Mean difference | Patient95% interval | All3seeds |
|---|---|---:|---|---|
| mixed | self_bootstrap vs baseline | 0.0103 | [-0.0440,0.0602] | False |
| mixed | soft_mode_control vs baseline | 0.0248 | [-0.0423,0.0694] | False |
| mixed | paired_union vs baseline | 0.1208 | [0.0021,0.2400] | False |
| mixed | paired_union vs soft_mode_control | 0.0959 | [0.0041,0.1871] | False |
| mixed | paired_stratified vs soft_mode_control | 0.0965 | [-0.0499,0.2384] | False |
| mixed | paired_stratified vs paired_union | 0.0006 | [-0.0767,0.0827] | False |
| from_SLR | self_bootstrap vs baseline | 0.0021 | [-0.1574,0.1169] | False |
| from_SLR | soft_mode_control vs baseline | -0.0647 | [-0.3860,0.1923] | False |
| from_SLR | paired_union vs baseline | -0.1573 | [-0.6845,0.6614] | False |
| from_SLR | paired_union vs soft_mode_control | -0.0927 | [-0.4486,0.4691] | False |
| from_SLR | paired_stratified vs soft_mode_control | -0.1219 | [-0.4622,0.4276] | False |
| from_SLR | paired_stratified vs paired_union | -0.0292 | [-0.0415,-0.0136] | False |
| from_ipod | self_bootstrap vs baseline | -0.0671 | [-0.1820,0.0155] | True |
| from_ipod | soft_mode_control vs baseline | 0.1786 | [-0.1784,0.4951] | False |
| from_ipod | paired_union vs baseline | -0.0299 | [-0.5152,0.2358] | False |
| from_ipod | paired_union vs soft_mode_control | -0.2085 | [-0.3368,0.0166] | False |
| from_ipod | paired_stratified vs soft_mode_control | -0.1901 | [-0.3178,0.0308] | False |
| from_ipod | paired_stratified vs paired_union | 0.0184 | [-0.0124,0.0535] | False |

![Risk and coverage](risk_coverage.png)

All six fixed coverages, p95/tails and full curves are retained. Ranking is
unmodified hypothesis dispersion, not a calibrated expected-error head or
a per-image guarantee. No matched calibrated C+ improvement is established.
Historical ordinary pair source means3.3680/4.8747/5.9025 and training-only
graph reverse4.9736 remain relevant stronger controls, with their different
inference costs explicitly recorded in prior reports.

## Separate known-source TRAIN routing diagnostic

Six frozen models process identical1,421virtual TRAIN bags. Known-mode gates
use the original capture labels, which are not provided at deployment.
These numbers are not image-model validation or an independent result.

| Source model | Learned global | Known global | Weighted known global | Known local | Shuffled local |
|---|---:|---:|---:|---:|---:|
| mixed__baseline__s17 | 3.9575 | 4.6686 | 4.6693 | 4.6734 | 4.6723 |
| mixed__baseline__s29 | 4.0116 | 5.1000 | 5.1256 | 5.0989 | 5.1048 |
| mixed__baseline__s43 | 4.3770 | 5.1368 | 5.1519 | 5.1506 | 5.1383 |
| mixed__paired_union__s17 | 3.5421 | 4.5388 | 4.7313 | 4.7114 | 4.5410 |
| mixed__paired_union__s29 | 3.4731 | 4.2153 | 4.3405 | 4.3280 | 4.2178 |
| mixed__paired_union__s43 | 3.5795 | 4.6140 | 4.7978 | 4.7802 | 4.6182 |

Supplying true patch-source mode does not rescue these frozen experts.
A color hypothesis trained only through a weighted sum is not automatically
an individually correct, physically identified conditional estimator. This
diagnostic does not rule out a differently trained local router. True Lab
was used only for scoring, not route selection. Unit counterexamples show
that varying gates and pooling need not commute, not that this causes the
real-data failure. Preserve this distinction.

## Compute and verification

All models contain929,297parameters; largest checkpoint3,722,227bytes.
Maximum fit-process GPU allocation108.51MiB.
Support changes are TRAIN-only; inference still takes one64-patch image bag.
No new batch1latency/inference VRAM or export claim is made.

135exact color arrays,45gate/hypothesis sets,
7920independent scalar color cases,270coverage rows and7920curve points.
276480observed patch tokens checked against independent explicit source/index construction.
Every epoch pair/plan digest and aggregate count replays; all same-site labels,
fit-only scales, shared initialization and same-camera selection are checked.
Routing diagnostic independently checks42630scalar color cases on identical inputs.
Original MSKCC CC-BY; no new external data or weights; no participant files
published. Independent result remains4.4570/80%4.1591 versus ordinary
fusion4.3005/4.1447. Product skin precision and universality remain unmet.
