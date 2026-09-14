# Luma ChromaSeed-G: a compact conditional color readout

**A modest mixed-role improvement, with an explicit camera-transfer limit.** The perceptual hard-gate model gives5.2851 ΔE00 versus5.3901 for its exact shared base (−0.1050, about1.95%) and5.3617 for a uniform correction. Active weights occupy21,973 numeric bytes. Its separate NumPy consumer responds in11.7µs; complete mixed-role fitting takes35.53ms on734 prepared rows. These are exploratory results from six held people and reused splits, not a validated phone-face product or a universal winner.

G follows the [D camera/support diagnostic](../chromaseed_camera_support_v1/report.md). It keeps one128-center kernel basis and adds a small residual whose sign/strength depends on current input features. Fit-side acquisition labels train the gate; query camera labels, measured Lab and query-distribution statistics are never inputs. The two known camera groups contain different people/acquisitions, so neither D nor G isolates a causal camera effect.

## All selected families and roles

Only original TRAIN:966 prepared skin-region records,24 people. Mixed fits734 rows/18 people and holds232/6; SLR→iPod fits323/8 and holds643/16; reverse swaps these. Three fixed inner person-disjoint folds choose settings for each family before any final-role score. All24 choices were frozen before all three final banks finished and before72 selected models were evaluated. Roles overlap and have been historically reused. Seeds17/29/43 are repeated fits: the table averages their errors, not their predictions, and does not count seeds as independent people.

| Family | Mixed ΔE00 ↓ | SLR → iPod ↓ | iPod → SLR ↓ | Mixed numeric bytes |
|---|---:|---:|---:|---:|
| MSE base | 5.438652 | 8.597000 | 8.705018 | 20,284 |
| MSE uniform | 5.384387 | 8.818465 | 8.742294 | 20,284 |
| MSE soft gate | 5.349017 | 8.597000 | 8.705018 | 21,973 |
| MSE hard gate | 5.322671 | 8.597000 | 8.705018 | 21,973 |
| Perceptual base | 5.390066 | 8.654805 | 8.386889 | 20,284 |
| Perceptual uniform | 5.361655 | 8.684220 | 8.585753 | 20,284 |
| Perceptual soft gate | 5.295118 | 8.654805 | 8.386889 | 21,973 |
| Perceptual hard gate | 5.285104 | 8.654805 | 8.386889 | 21,973 |

All routed transfer choices return the exact base because their fit side contains one camera group. This is a fit-time fallback, not detection of an unknown camera during deployment. Uniform residual correction remains active and worsens both transfer directions for both bases. G therefore adds no demonstrated unseen-camera benefit.

The matching base payloads reproduce [P](../chromaseed_perceptual_v1/report.md) exactly. Earlier strong references remain relevant: [full KRR](../skin_local_search_v1/report.md) gives5.3984/8.5845/8.9125 with larger, role-dependent storage; the compact guided RBF gives7.7193 in reverse, better than G's8.3869. [R's dynamic model](../chromaseed_refine_v1/report.md) gives5.3551/9.9089/9.8122 at59,316B. G's mixed result is descriptively lower, but no fresh outer sample selects or confirms a final family.

![Quality and runtime](gated_results.png)

## Selection and secondary metrics

The primary score first averages native ΔE00 across images of each person, then across people. Training instead weights people, sites and their images equally. Site-balanced and image-tail summaries are retained below; p90 is the mean of per-seed image p90 values. Lambda10 with rho0 is a zero-alias placeholder, not a fitted residual penalty.

| Role | Family | λ / ρ | Inner person ΔE00 | Held image mean | Held site/person mean | Held image p90 |
|---|---|---:|---:|---:|---:|---:|
| Mixed | MSE base | 10 / 0 | 4.455353 | 5.6221 | 5.4572 | 10.3506 |
| Mixed | MSE uniform | 0.1 / 0.5 | 4.442135 | 5.5558 | 5.4003 | 10.2425 |
| Mixed | MSE soft gate | 0.1 / 0.5 | 4.429678 | 5.5213 | 5.3670 | 10.0958 |
| Mixed | MSE hard gate | 0.1 / 0.5 | 4.413827 | 5.4948 | 5.3390 | 9.9906 |
| Mixed | Perceptual base | 10 / 0 | 4.448451 | 5.5736 | 5.4102 | 10.1994 |
| Mixed | Perceptual uniform | 1 / 1 | 4.438933 | 5.5379 | 5.3808 | 10.1299 |
| Mixed | Perceptual soft gate | 0.1 / 0.5 | 4.431187 | 5.4690 | 5.3139 | 9.9522 |
| Mixed | Perceptual hard gate | 0.1 / 0.5 | 4.420066 | 5.4567 | 5.3023 | 10.0012 |
| SLR → iPod | MSE base | 10 / 0 | 5.058676 | 8.8450 | 8.6149 | 13.6453 |
| SLR → iPod | MSE uniform | 0.1 / 1 | 4.878367 | 9.0389 | 8.8410 | 13.8970 |
| SLR → iPod | MSE soft gate | 10 / 0 | 5.058676 | 8.8450 | 8.6149 | 13.6453 |
| SLR → iPod | MSE hard gate | 10 / 0 | 5.058676 | 8.8450 | 8.6149 | 13.6453 |
| SLR → iPod | Perceptual base | 10 / 0 | 5.177163 | 8.9011 | 8.6748 | 13.6978 |
| SLR → iPod | Perceptual uniform | 0.1 / 1 | 5.067245 | 8.8698 | 8.7072 | 13.6099 |
| SLR → iPod | Perceptual soft gate | 10 / 0 | 5.177163 | 8.9011 | 8.6748 | 13.6978 |
| SLR → iPod | Perceptual hard gate | 10 / 0 | 5.177163 | 8.9011 | 8.6748 | 13.6978 |
| iPod → SLR | MSE base | 10 / 0 | 4.371878 | 8.7759 | 8.7316 | 15.5016 |
| iPod → SLR | MSE uniform | 0.1 / 0.5 | 4.364153 | 8.7970 | 8.7647 | 15.2637 |
| iPod → SLR | MSE soft gate | 10 / 0 | 4.371878 | 8.7759 | 8.7316 | 15.5016 |
| iPod → SLR | MSE hard gate | 10 / 0 | 4.371878 | 8.7759 | 8.7316 | 15.5016 |
| iPod → SLR | Perceptual base | 10 / 0 | 4.332083 | 8.4466 | 8.4143 | 15.0304 |
| iPod → SLR | Perceptual uniform | 0.1 / 1 | 4.308546 | 8.6224 | 8.6032 | 15.2084 |
| iPod → SLR | Perceptual soft gate | 10 / 0 | 4.332083 | 8.4466 | 8.4143 | 15.0304 |
| iPod → SLR | Perceptual hard gate | 10 / 0 | 4.332083 | 8.4466 | 8.4143 | 15.0304 |

## Paired mixed-role checks

For each person, average the three seed-specific errors before forming paired differences. Fixed-prediction bootstrap uses20,000 shared resamples of six people. The following95% ranges are descriptive: they omit fit/selection uncertainty and repeated research on these same people. They are not confirmatory population intervals or significance claims.

| Candidate | Matched reference | Mean difference ↓ | People improved | Descriptive range |
|---|---|---:|---:|---|
| MSE uniform | MSE base | -0.054265 | 4/6 | [-0.134233, +0.020676] |
| MSE soft gate | MSE base | -0.089635 | 4/6 | [-0.239117, +0.016089] |
| MSE soft gate | MSE uniform | -0.035371 | 3/6 | [-0.124019, +0.031529] |
| MSE hard gate | MSE base | -0.115981 | 4/6 | [-0.256804, -0.000754] |
| MSE hard gate | MSE uniform | -0.061716 | 5/6 | [-0.132460, -0.005364] |
| Perceptual uniform | Perceptual base | -0.028411 | 3/6 | [-0.077192, +0.018145] |
| Perceptual soft gate | Perceptual base | -0.094948 | 5/6 | [-0.235523, +0.005784] |
| Perceptual soft gate | Perceptual uniform | -0.066537 | 4/6 | [-0.181396, +0.012361] |
| Perceptual hard gate | Perceptual base | -0.104962 | 4/6 | [-0.229843, +0.002567] |
| Perceptual hard gate | Perceptual uniform | -0.076551 | 5/6 | [-0.164658, -0.006365] |

Perceptual hard improves four of six people against its base; its descriptive range includes zero. Soft improves five of six with a very similar mean. Hard routing has a discontinuity at gate score zero; observed ordinary-input quality alone does not establish its stability under color perturbations. That is the next bounded diagnostic, with fixed weights and prespecified perturbations, not another outer-tuned grid.

## Actual deployment and training cost

One CPU thread on AMD Ryzen9 7900X, Windows11, NumPy2.5.3. The NumPy-only consumer imports neither Torch nor SciPy and caches converted arrays once. Both implementations were checked on every selected query using actual batch-one calls;20 warmups and3 full query passes per model. Latencies below are medians across three per-seed medians; p95 is the median across seed p95 values. Complete fitting is seed17, three timed fits after one warmup, with every resulting array exactly equal to its frozen payload. No concurrent heavy study process ran during profiling.

| Role | Family | Numeric B | Canonical µs | NumPy µs / p95 | Cached array B | Full fit ms |
|---|---|---:|---:|---:|---:|---:|
| Mixed | MSE base | 20,284 | 16.50 | 9.30 / 9.50 | 41,272 | 16.65 |
| Mixed | MSE uniform | 20,284 | 16.50 | 9.30 / 9.50 | 41,272 | 18.43 |
| Mixed | MSE soft gate | 21,973 | 22.60 | 11.80 / 12.00 | 44,640 | 28.05 |
| Mixed | MSE hard gate | 21,973 | 22.40 | 11.80 / 12.00 | 44,640 | 28.15 |
| Mixed | Perceptual base | 20,284 | 16.60 | 9.40 / 9.50 | 41,272 | 22.98 |
| Mixed | Perceptual uniform | 20,284 | 16.50 | 9.40 / 9.60 | 41,272 | 24.79 |
| Mixed | Perceptual soft gate | 21,973 | 22.80 | 11.80 / 12.00 | 44,640 | 34.51 |
| Mixed | Perceptual hard gate | 21,973 | 22.40 | 11.70 / 11.90 | 44,640 | 35.53 |
| SLR → iPod | MSE base | 20,284 | 16.40 | 9.30 / 9.50 | 41,272 | 8.48 |
| SLR → iPod | MSE uniform | 20,284 | 16.60 | 9.40 / 9.60 | 41,272 | 10.13 |
| SLR → iPod | MSE soft gate | 20,284 | 16.50 | 9.40 / 9.50 | 41,272 | 8.52 |
| SLR → iPod | MSE hard gate | 20,284 | 16.50 | 9.50 / 9.70 | 41,272 | 8.52 |
| SLR → iPod | Perceptual base | 20,284 | 16.80 | 9.60 / 9.70 | 41,272 | 12.31 |
| SLR → iPod | Perceptual uniform | 20,284 | 16.80 | 9.60 / 9.90 | 41,272 | 13.48 |
| SLR → iPod | Perceptual soft gate | 20,284 | 16.90 | 9.60 / 9.90 | 41,272 | 12.19 |
| SLR → iPod | Perceptual hard gate | 20,284 | 16.90 | 9.60 / 9.80 | 41,272 | 12.20 |
| iPod → SLR | MSE base | 20,284 | 16.80 | 9.60 / 9.80 | 41,272 | 14.61 |
| iPod → SLR | MSE uniform | 20,284 | 16.90 | 9.60 / 9.70 | 41,272 | 16.39 |
| iPod → SLR | MSE soft gate | 20,284 | 16.90 | 9.60 / 9.70 | 41,272 | 14.60 |
| iPod → SLR | MSE hard gate | 20,284 | 16.90 | 9.60 / 9.80 | 41,272 | 14.53 |
| iPod → SLR | Perceptual base | 20,284 | 16.90 | 9.60 / 9.70 | 41,272 | 20.27 |
| iPod → SLR | Perceptual uniform | 20,284 | 16.90 | 9.50 / 9.70 | 41,272 | 21.87 |
| iPod → SLR | Perceptual soft gate | 20,284 | 16.90 | 9.60 / 9.80 | 41,272 | 20.35 |
| iPod → SLR | Perceptual hard gate | 20,284 | 16.90 | 9.50 / 9.90 | 41,272 | 20.30 |

The active gate adds1,689 numeric bytes (8.33%) over20,284B; the uniform correction collapses into existing coefficients. The standalone predictor caches44,640B of arrays for an active model versus41,272B for a static model. These exclude Python/NumPy/process memory, temporary arrays, retained input dictionaries and compressed archive overhead;21,973B is not end-to-end RAM. Predictor construction and validation are timed separately in runtime.json; archive loading is excluded. Image decoding, skin localization, color36 extraction, network transport and mobile hardware are outside all quoted latencies.

Four fixed active probes (both bases × soft/hard, λ0.1/ρ1) ensure active-path costs are measured even when a selected family elsewhere falls back. They add16 full fits including warmups to96 selected fits; no new quality choice was made from these probes.

| Fixed active mixed probe | NumPy µs | Canonical µs | Full fit ms |
|---|---:|---:|---:|
| MSE soft gate | 12.20 | 23.50 | 28.55 |
| MSE hard gate | 12.00 | 23.20 | 28.06 |
| Perceptual soft gate | 12.10 | 23.40 | 34.39 |
| Perceptual hard gate | 12.10 | 23.10 | 34.27 |

## What was fitted and independently checked

The primary workflow took6.671s including reading, fitting, selection, persistence and evaluation, excluding interpreter imports, audit and profiling. Across12 banks:2,016 stored readouts include864 exact one-camera routed fallbacks and72 unchanged P base controls. Only360 positive residual coefficient solutions were solved, sharing120 algebra operations across three penalties; there were36 perceptual base solves and4 gate fits. Three strengths reuse each coefficient solution. Do not call2,016 configurations2,016 independent trainings.

Residual fitting uses an analytic weighted ridge solution in the existing whitened kernel basis. It localizes a correction in the readout without global backpropagation or brute-force weight search. The current input modulates effective output coefficients; the model does not learn at query time, add connections, or repeatedly run until convergence. All360 regularized residual objectives are nonincreasing, which is a fit check rather than evidence of generalization.

Primary23 numerical tests passed in1.56s; three separate portable-consumer tests passed in0.25s. The audit checks all12 banks/2,016 readouts,72 exact P controls,864 fallbacks,285,600 OOF query rows,24 choices and72 selected models/28,752 final query rows. All72 selected models were independently refitted using direct kernels, analytic perceptual tensors and augmented SVD;12 extra positive-route probes passed. Maximum independent-refit native-Lab drift 2.46e-05, versus registered0.001 tolerance. Direct OOF/final drift is below2e-12. The independent audit shares the existing verified ΔE00 formula and frozen split helpers.

Primary PID36812, audit session27067 and runtime returned exit0. Source locks bind the core, runner, primary tests/protocol, inherited dependencies, all P control banks and D verification. The NumPy consumer, portable tests and postprocessing are additionally bound by final verification. Earlier locked sources and results are preserved.

## Product boundary and next decision

Luma ChromaSeed is the working family name; G denotes this conditional-readout experiment. Input is36 statistics from an already prepared skin region; output is three native Lab values in the dataset's D65/10° convention. The model does not detect faces, identify people, infer ethnicity, diagnose skin, calibrate an arbitrary camera, or establish cosmetic shade-match accuracy. No real ordinary-phone face benchmark or retailer integration is demonstrated. Dataset/weight commercial-use clearance is not inferred from a code implementation.

Keep the base and both gates as research candidates. Next measure gate-score margin and output sensitivity to fixed, small synthetic color transformations, using the frozen payloads and no retraining. Report hard/soft/base controls and discontinuities explicitly; synthetic robustness cannot replace independent face acquisition. The broader compact/fast/high-quality goal remains active.

Conditional expert readouts are established ideas: [Jacobs et al.,1991](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) and feature modulation in [Perez et al.,FiLM](https://arxiv.org/abs/1709.07871). This experiment makes no invention or breakthrough claim. No new data, model weights or third-party code were downloaded; only the original TRAIN arrays were accessed, with legacy validation/calibration/test excluded.

[Protocol](../../research/chromaseed_gated_v1_protocol.md) · [Reproduce](reproduce.md) · [Audit](audit.json) · [Runtime](runtime.json) · [Summary](summary.json) · [Verification](verification.json) · [Model card](../../architecture/chromaseed_gated_model_card.md) · [Next decision](../../research/chromaseed_gated_next_decision.md)
