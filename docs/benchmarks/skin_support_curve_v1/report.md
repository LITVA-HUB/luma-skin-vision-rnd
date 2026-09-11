# Actual skin color: TRAIN-person support and representation screen

27 locally reproduced fits on original public MSKCC images and native instrument Lab.
Only original TRAIN was loaded. Fixed internal holdout: six people, 232 images.
Nested training subsets: 6/12/18 people. All fits use exactly 930 optimizer updates.
These are exploratory internal results, not the archived independent test.
Reported groups average three individual runs, not ensemble predictions.

| People / model | Training mean | Holdout mean | Median | p95 | At 80% |
|---|---:|---:|---:|---:|---:|
| 6/baseline | 3.0095 | 7.2439 | 5.7243 | 17.8575 | 6.1440 |
| 6/statistics | 2.9652 | 7.2068 | 5.6289 | 17.7882 | 6.1219 |
| 6/pixels | 2.9945 | 7.1762 | 5.6432 | 17.7771 | 6.1150 |
| 12/baseline | 3.4065 | 6.2843 | 5.1261 | 15.0596 | 5.5917 |
| 12/statistics | 3.3636 | 6.2940 | 5.1346 | 15.6009 | 5.5743 |
| 12/pixels | 3.3895 | 6.2744 | 5.1056 | 15.2808 | 5.5750 |
| 18/baseline | 3.5823 | 6.1082 | 5.1353 | 14.9960 | 5.3918 |
| 18/statistics | 3.5366 | 6.1560 | 5.1727 | 15.1242 | 5.4256 |
| 18/pixels | 3.5742 | 6.1258 | 5.1768 | 15.0828 | 5.4081 |

More training people help this fixed-budget screen. Learned pixel residuals do
not provide a convincing advantage over the matched statistics residual or original
baseline. This does not prove data size is the only bottleneck or that other pixel
models cannot work. These budgets/populations differ from historical 80-epoch fits.

![Learning and selection curves](learning_curve.png)

At a fixed person count and seed, all models share identical input-novelty accept
sets. Across person counts these sets may differ. Novelty is not calibrated color
error; the generic dispersion-MAE field is not a reliability guarantee. All six
fixed coverages are in per-fit results; full curves are in risk_coverage.csv.

## Matched descriptive differences

Negative favors candidate. Patient intervals use six clusters and 10,000 bootstrap
draws after averaging model-seed errors. They describe patient-balanced differences,
not image-weighted differences, and are not multiplicity-adjusted confirmation.

| People | Candidate vs control | Image difference | Patient difference | Patient interval |
|---:|---|---:|---:|---|
| 6 | pixels vs statistics | -0.0305 | -0.0022 | [-0.1123, 0.1230] |
| 6 | pixels vs baseline | -0.0677 | -0.0872 | [-0.1966, 0.0069] |
| 6 | statistics vs baseline | -0.0371 | -0.0850 | [-0.2956, 0.0443] |
| 12 | pixels vs statistics | -0.0196 | -0.0139 | [-0.0522, 0.0245] |
| 12 | pixels vs baseline | -0.0100 | -0.0128 | [-0.0395, 0.0173] |
| 12 | statistics vs baseline | 0.0096 | 0.0011 | [-0.0569, 0.0608] |
| 18 | pixels vs statistics | -0.0302 | -0.0176 | [-0.0702, 0.0389] |
| 18 | pixels vs baseline | 0.0176 | 0.0136 | [-0.0045, 0.0309] |
| 18 | statistics vs baseline | 0.0478 | 0.0311 | [-0.0432, 0.0953] |

## Frozen branch and pixel-information diagnostics

Removing a trained residual is an intervention, not a retrained baseline. RGB
permutation moves triples within each patch; original statistical tokens remain
untouched. Mean RGB removes within-patch variation from the added encoder only.

| People / model | Intervention | Mean skin error | Prediction Lab RMS change |
|---|---|---:|---:|
| 6/statistics | zero | 7.9942 | 2.1944 |
| 6/pixels | zero | 7.6479 | 1.2816 |
| 6/pixels | shuffle | 7.1756 | 0.0373 |
| 6/pixels | mean | 7.1695 | 0.0322 |
| 12/statistics | zero | 7.6113 | 2.5550 |
| 12/pixels | zero | 6.8297 | 1.3601 |
| 12/pixels | shuffle | 6.2706 | 0.0267 |
| 12/pixels | mean | 6.2654 | 0.0214 |
| 18/statistics | zero | 7.3067 | 2.1069 |
| 18/pixels | zero | 6.6175 | 0.9890 |
| 18/pixels | shuffle | 6.1266 | 0.0175 |
| 18/pixels | mean | 6.1199 | 0.0160 |

Removing adapters degrades their trained models. Yet shuffling or replacing patch
pixels by their means barely changes the pixel model. Its residual is active,
but useful spatial structure has not been demonstrated. Existing statistical
tokens already contain the patch mean; a branch can be necessary to its trained
parameterization without contributing new information or better final accuracy.

Reference-palette nearest-color mean distance decreases 2.5713 -> 2.0602 -> 1.6696
as training people increase. This uses actual held-out reference colors and is
strictly a support diagnostic, not image inference, a noise floor or a risk score.

## Reproducibility and scope

27 exact checkpoint prediction arrays; 3 complete exact state refits; 6264 scalar color cases; 162 fixed-coverage rows.
784392 independent reference-support distances; 3456 source patch layout checks.
Diagnostics add 27 training arrays and 36 intervention arrays, 21426 scalar color cases and 267264 patch provenance checks.
Parameters by model: {'baseline': 929297, 'statistics': 940491, 'pixels': 940763}.
Maximum fit allocation 318.36 MiB; largest checkpoint 3,770,267 bytes. No new isolated latency/export measurement.
No test-camera holdout here. Camera identity is not an input; both acquisition
families are represented in training. Within each draw, camera proportions are
fixed. Participant counts and skin-color support still co-vary.

Original dataset CC-BY; no new data/weights, cloud resources or publication.
Independent MSKCC remains primary4.4570/80%4.1591 vs ordinary fusion4.3005/4.1447.
No demonstrated universal smartphone/facial accuracy or proposed novelty win.
[Research decision](../../research/skin_support_curve_next_decision.md).
