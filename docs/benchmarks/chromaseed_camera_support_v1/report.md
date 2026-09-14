# ChromaSeed-D: camera and target-support diagnosis

**Progress toward a justified adaptive color model; no new skin-color accuracy claim.** A fixed linear classifier distinguishes all 24 person-held-out camera labels from color36. AUC from native instrument Lab alone is only0.586 (linear) and0.578 (RBF). Camera separation persists after a fit-only linear Lab residualization. This supports testing an acquisition-conditioned color correction, while different people/acquisition per camera prevent a causal conclusion.

The previous [S study](../chromaseed_selection_stability_v1/report.md) showed selection instability but did not explain the transfer loss. D follows its registered next question. It uses original TRAIN only and does not access prior outer model/results/prediction archives.

## Fixed camera classifiers: all views and methods

Three person-disjoint folds over24 people (8 SLR,16 iPod),966 prepared skin-region records. Equal camera/person/site/image fit weights. Each person gets one score averaged equally across their sites. Linear alpha0.1 and RBF alpha0.01 are fixed; no search or threshold tuning. Oracle controls consume measured target Lab and cannot be deployed on an unmeasured selfie.

| Input | Method | Person AUC | Balanced accuracy, threshold0 | Correct SLR people | Correct iPod people | Fold AUCs |
|---|---|---:|---:|---:|---:|---|
| All 36 color features | linear | 1.0000 | 1.0000 | 8/8 | 16/16 | 1.0000, 1.0000, 1.0000 |
| All 36 color features | rbf | 1.0000 | 0.9375 | 7/8 | 16/16 | 1.0000, 1.0000, 1.0000 |
| Mean RGB only | linear | 0.6484 | 0.6875 | 4/8 | 14/16 | 1.0000, 0.4667, 0.8000 |
| Mean RGB only | rbf | 0.9844 | 0.7500 | 4/8 | 16/16 | 1.0000, 1.0000, 1.0000 |
| Instrument Lab (oracle) | linear | 0.5859 | 0.6250 | 5/8 | 10/16 | 0.7778, 0.4000, 0.6000 |
| Instrument Lab (oracle) | rbf | 0.5781 | 0.5938 | 4/8 | 11/16 | 0.6667, 0.4000, 0.5000 |
| Color residual after Lab (oracle) | linear | 1.0000 | 0.9688 | 8/8 | 15/16 | 1.0000, 1.0000, 1.0000 |
| Color residual after Lab (oracle) | rbf | 1.0000 | 0.9688 | 8/8 | 15/16 | 1.0000, 1.0000, 1.0000 |

AUC measures ranking of camera scores, not correctness of predicted skin color. A high AUC can coexist with threshold errors: RBF color36 ranks all people correctly but misses one SLR person at the fixed zero threshold. Fold AUCs and pooled AUC differ when independently fitted fold score scales differ. No population confidence intervals or causal tests are claimed.

![Camera signal and coverage](camera_support.png)

## Target-color matching within held folds

Pair people using equal-site native-Lab centroids, maximizing pair count under each prespecified DeltaE00 caliper and then minimizing total distance. Every person is used at most once per caliper. All pairings stay within the same held fold. Matching conditions evaluation on held targets without fitting classifiers on them; it does not equalize within-person color distributions or other acquisition differences.

| Caliper | Pairs | Matched people | Unmatched SLR / iPod | Matched images | Mean pair DeltaE00 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 8 / 16 | 0 | — |
| 2 | 2 | 4 | 6 / 14 | 176 | 1.9005 |
| 3 | 2 | 4 | 6 / 14 | 176 | 1.9005 |
| 5 | 4 | 8 | 4 / 12 | 323 | 2.7038 |
| 10 | 8 | 16 | 0 / 8 | 640 | 5.0722 |

At calipers2 and3 only two pairs/four people are represented. Perfect camera ranking in that subset is therefore limited evidence. All calipers are reported; none was chosen for its favorable ranking.

| Caliper | Input | Method | Matched-person AUC | Balanced accuracy | SLR score above paired iPod (ties half) |
|---:|---|---|---:|---:|---:|
| 1 | All 36 color features | linear | — | — | — |
| 1 | All 36 color features | rbf | — | — | — |
| 1 | Mean RGB only | linear | — | — | — |
| 1 | Mean RGB only | rbf | — | — | — |
| 1 | Instrument Lab (oracle) | linear | — | — | — |
| 1 | Instrument Lab (oracle) | rbf | — | — | — |
| 1 | Color residual after Lab (oracle) | linear | — | — | — |
| 1 | Color residual after Lab (oracle) | rbf | — | — | — |
| 2 | All 36 color features | linear | 1.0000 | 1.0000 | 1.0000 |
| 2 | All 36 color features | rbf | 1.0000 | 0.7500 | 1.0000 |
| 2 | Mean RGB only | linear | 0.5000 | 0.5000 | 0.5000 |
| 2 | Mean RGB only | rbf | 1.0000 | 0.5000 | 1.0000 |
| 2 | Instrument Lab (oracle) | linear | 0.2500 | 0.5000 | 0.0000 |
| 2 | Instrument Lab (oracle) | rbf | 0.2500 | 0.5000 | 0.0000 |
| 2 | Color residual after Lab (oracle) | linear | 1.0000 | 0.7500 | 1.0000 |
| 2 | Color residual after Lab (oracle) | rbf | 1.0000 | 1.0000 | 1.0000 |
| 3 | All 36 color features | linear | 1.0000 | 1.0000 | 1.0000 |
| 3 | All 36 color features | rbf | 1.0000 | 0.7500 | 1.0000 |
| 3 | Mean RGB only | linear | 0.5000 | 0.5000 | 0.5000 |
| 3 | Mean RGB only | rbf | 1.0000 | 0.5000 | 1.0000 |
| 3 | Instrument Lab (oracle) | linear | 0.2500 | 0.5000 | 0.0000 |
| 3 | Instrument Lab (oracle) | rbf | 0.2500 | 0.5000 | 0.0000 |
| 3 | Color residual after Lab (oracle) | linear | 1.0000 | 0.7500 | 1.0000 |
| 3 | Color residual after Lab (oracle) | rbf | 1.0000 | 1.0000 | 1.0000 |
| 5 | All 36 color features | linear | 1.0000 | 1.0000 | 1.0000 |
| 5 | All 36 color features | rbf | 1.0000 | 0.8750 | 1.0000 |
| 5 | Mean RGB only | linear | 0.6875 | 0.6250 | 0.7500 |
| 5 | Mean RGB only | rbf | 1.0000 | 0.7500 | 1.0000 |
| 5 | Instrument Lab (oracle) | linear | 0.3125 | 0.5000 | 0.0000 |
| 5 | Instrument Lab (oracle) | rbf | 0.3750 | 0.5000 | 0.0000 |
| 5 | Color residual after Lab (oracle) | linear | 1.0000 | 0.8750 | 1.0000 |
| 5 | Color residual after Lab (oracle) | rbf | 1.0000 | 1.0000 | 1.0000 |
| 10 | All 36 color features | linear | 1.0000 | 1.0000 | 1.0000 |
| 10 | All 36 color features | rbf | 1.0000 | 0.9375 | 1.0000 |
| 10 | Mean RGB only | linear | 0.6719 | 0.6250 | 0.6250 |
| 10 | Mean RGB only | rbf | 0.9844 | 0.7500 | 1.0000 |
| 10 | Instrument Lab (oracle) | linear | 0.5156 | 0.5625 | 0.5000 |
| 10 | Instrument Lab (oracle) | rbf | 0.5312 | 0.5625 | 0.2500 |
| 10 | Color residual after Lab (oracle) | linear | 1.0000 | 0.9375 | 1.0000 |
| 10 | Color residual after Lab (oracle) | rbf | 1.0000 | 0.9375 | 1.0000 |

## Coverage of the historical transfer roles

Reference distances for fit rows exclude every row from their own person. Color36 uses fit-only weighted standardization and RMS distances; native Lab uses DeltaE00. Radius is the fit reference weighted empirical95th percentile. Query mass weights people and their sites equally. This is descriptive feature geometry, not a calibrated rejection rule or error bound.

| Role | Space | Fit reference mean / p95 | Query mean / p95 | Query mass above fit radius |
|---|---|---:|---:|---:|
| Mixed cameras | color36 | 0.2422 / 0.5030 | 0.2413 / 0.4245 | 1.52% |
| Mixed cameras | native_lab | 1.5436 / 3.0313 | 1.6295 / 3.2837 | 10.61% |
| SLR → iPod | color36 | 0.2251 / 0.3964 | 0.3685 / 0.7860 | 28.31% |
| SLR → iPod | native_lab | 2.4792 / 6.7486 | 2.5410 / 6.2025 | 3.98% |
| iPod → SLR | color36 | 0.2819 / 0.5171 | 0.8236 / 2.3322 | 54.20% |
| iPod → SLR | native_lab | 1.6669 / 3.5974 | 1.8722 / 3.9588 | 9.34% |

Feature mass beyond the fit radius is28.31% for SLR→iPod and54.20% for iPod→SLR, versus1.52% in the mixed role. The corresponding native-Lab masses are3.98%,9.34% and10.61%. The two spaces have different geometries and radii, so their distances are not directly comparable. Still, the feature coverage pattern supports investigating acquisition-dependent transformations before another larger penalty/step search.

## Measured target coverage

| Camera | People / rows | Lab 5th percentile | Lab median | Lab 95th percentile |
|---|---:|---|---|---|
| SLR | 8 / 323 | (32.67, 3.33, 9.00) | (50.00, 9.00, 18.00) | (67.00, 13.33, 24.33) |
| ipod | 16 / 643 | (32.00, 1.00, 6.67) | (60.00, 7.67, 16.33) | (72.00, 14.00, 23.00) |

These are weighted component quantiles, not joint color samples or demographic categories. Native-Lab distributions do differ; the low target-only AUC does not prove identical populations. Person centroid extrema are retained in the machine-readable summary without identifiers.

## Decision and evidence boundary

Register a compact shared-kernel model with an acquisition-conditioned residual readout as the next experiment. Fit the gate and residual only on fit-side people, keep a shared-model control, and use exact shared fallback when a fit split contains only one camera group. Share the128-center basis so conditional coefficients add little storage and no second feature-extraction pass. This tests the requested dynamic connections in a bounded way; it does not assume that learning two known camera groups solves unseen-phone transfer. The next series is planned, not launched.

Do not use oracle Lab-residual features in a deployable predictor, import held-camera labels into training, pick a caliper/radius by outer error, or claim that a camera classifier proves useful skin-color correction. Compactness and approximately0.017ms prepared-feature inference are prior measurements of the earlier model, not results of these diagnostic classifiers. Real ordinary-phone face quality remains unvalidated, so the full user goal remains active.

## Verification

Eleven numerical tests pass. All24 classifiers were independently reconstructed with separate weighted normalizers and SVD solvers; all7728 query scores,8 pooled/24 fold metrics,15 exhaustive matching problems,40 matched metric groups,6 support cases/5796 row distances and2 target summaries passed. Maximum classifier-score drift 4.53e-14, support-distance drift 8.88e-16. The audit shares the existing verified CIEDE2000 formula and frozen folds.

Primary diagnostic time 0.688s excludes interpreter imports and auditing and is not a production training benchmark. Primary execution and final audit returned exit0. The first audit completed numerical checks but failed to serialize a NumPy integer counter; explicit scalar conversion fixed the report writer and the full audit was rerun successfully. No prior source lock was changed.

Only TRAIN color/target/patient/site/device arrays were loaded. No images, tokens, legacy validation/calibration/test, prior outer results, external downloads or publication. Stored predictions and pairings use ordinal axes; no direct patient identifiers.

Related classifier-based distribution diagnostics: [Jang et al. (2022)](https://proceedings.mlr.press/v162/jang22a.html). D is not their sequential hypothesis test. Numerical references: [SciPy assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html), [scikit-learn AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

[Protocol](../../research/chromaseed_camera_support_v1_protocol.md) · [Reproduce](reproduce.md) · [Audit](audit.json) · [Verification](verification.json) · [Summary](summary.json) · [Next decision](../../research/chromaseed_camera_support_next_decision.md)
