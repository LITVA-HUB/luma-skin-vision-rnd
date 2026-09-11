# Independent instrument-referenced skin-color test

All results below are REPRODUCED LOCALLY on real original MSKCC skin photographs and native Lab instrument measurements. Lower DeltaE00 is better. This is not illuminant angular error.

## Population and integrity

24 TRAIN people / 966 images; 6 source-VALIDATION people / 264 images; 6 CALIBRATION people / 208 images; 10 independent TEST people / 400 images / 105 sites. Person roles are disjoint. Both camera families were seen during development: this is a known-camera, new-person test.

Original source: [MSKCC release](https://api.isic-archive.com/doi/mskcc-skin-tone-labeling-dataset/), DOI 10.34970/962049, attribution Memorial Sloan Kettering Cancer Center. Original release/per-image metadata says CC-BY; no unsupported version number is assigned. 1,838 paired JPEGs, 2,128,062,766 bytes acquired and individually verified. Dataset rights do not substitute for a clinical/product validation.

Target: mean of actually published complete native Lab instrument readings (SkinColorCatch D65/10-degree convention). It is a site reference, not a dense per-pixel color map. The [pre-test amendment](../../research/skin_mskcc_reference_amendment_v1.md) records three calibration image rows with only one reading. No image was excluded or measurement imputed. Test images by number of complete readings: {"1": 0, "2": 0, "3": 400}.

The pre-test Git checkpoint is `checkpoint/skin-selective-pretest-2026-09-11`. Final lock SHA256: `8d30596335c20d6d1e57e4acf1994c16983580f76740fe7d2ea36d8feae94d6e`. All color/risk models, selections and calibration rules were fixed before TEST decoding. The exposed test is now an evaluation archive; it must not become a tuning set for a new claimed independent result.

## Main findings

The primary patch ensemble has mean **4.4570**, median **4.0012**, p95 **9.2631 DeltaE00** at full coverage. C+ and Proposed have exactly identical color predictions; they differ only in error ranking.

At exact 80% accepted coverage, Proposed mean **4.1591** versus C+ **4.3328**; difference **-0.1736**. The patient-cluster 95% bootstrap interval is **[-0.5684831772599341, 0.16472377164102991]**, crossing zero. This is a small observed gain, not convincing evidence of a novel mechanism.

The stronger simple density ranking on the same color predictions reaches **4.2770** at 80%. A frozen ordinary CNN/patch six-model fusion achieves full mean **4.3005** and density-ranked 80% mean **4.1447**. Proposed does NOT beat the strongest observed full-system comparator. The fusion has 7,337,589 color parameters, so it is not the 2.77M-parameter primary method.

## Fixed-coverage color risk

| Accepted | C+ mean | Proposed mean | Proposed median | Proposed p95 | Proposed error >10 |
|---:|---:|---:|---:|---:|---:|
| 100% | 4.4570 | 4.4570 | 4.0012 | 9.2631 | 3.75% |
| 95% | 4.4452 | 4.3176 | 3.9625 | 8.8808 | 3.16% |
| 90% | 4.4401 | 4.2439 | 3.9323 | 8.5672 | 2.78% |
| 80% | 4.3328 | 4.1591 | 3.8834 | 8.0966 | 1.88% |
| 70% | 4.3037 | 4.1219 | 3.8834 | 8.2565 | 1.79% |
| 60% | 4.1801 | 4.1287 | 3.9117 | 7.9873 | 1.67% |

Exact-coverage ranking is not a deployed threshold guarantee. The threshold set at 80% CALIBRATION coverage accepts **74.25%** of TEST images with mean **4.1486**. A calibrated predicted-error <=2 threshold accepts zero test images; <=5 accepts 69.25%, and 29.24% of those still exceed actual DeltaE00 5. Expected error is not a per-image upper bound. Test predicted-error MAE is 2.0543.

![Risk versus coverage](risk_coverage.png)

## All frozen color comparators

Sorted by observed test mean for descriptive reporting only; no post-test fitting or selection is performed. Their 80% columns all use the same TRAIN-density ranking.

| Locally reproduced method | Mean | Median | p95 | Mean at 80% |
|---|---:|---:|---:|---:|
| fusion_ensemble | 4.3005 | 3.9637 | 8.6024 | 4.1447 |
| fusion_s43 | 4.3162 | 4.0815 | 8.3378 | 4.2093 |
| shared_tight_ensemble | 4.4188 | 3.9121 | 9.2333 | 4.2269 |
| fusion_s29 | 4.4334 | 3.9627 | 8.8522 | 4.2993 |
| cnn_ensemble | 4.4481 | 4.1044 | 8.5979 | 4.3071 |
| votes_huber3_ensemble | 4.4521 | 4.0130 | 9.2261 | 4.2718 |
| votes_mean_ensemble | 4.4570 | 4.0012 | 9.2631 | 4.2770 |
| votes_huber3_s17 | 4.4595 | 4.1001 | 8.9004 | 4.2941 |
| shared_tight_s17 | 4.4599 | 4.0937 | 8.8132 | 4.2946 |
| votes_mean_s17 | 4.4667 | 4.1037 | 9.0071 | 4.3013 |
| votes_huber3_s29 | 4.4936 | 3.9777 | 9.7681 | 4.2959 |
| fusion_s17 | 4.4957 | 4.1849 | 9.5850 | 4.2456 |
| shared_tight_s43 | 4.4962 | 3.9614 | 9.3187 | 4.3156 |
| votes_mean_s29 | 4.4975 | 4.0066 | 9.8370 | 4.3029 |
| votes_huber3_s43 | 4.5072 | 3.9585 | 9.4560 | 4.3237 |
| votes_mean_s43 | 4.5096 | 3.9778 | 9.4858 | 4.3260 |
| global_mlp_s43 | 4.5555 | 4.1031 | 9.4958 | 4.2454 |
| global_mlp_ensemble | 4.5804 | 4.1064 | 10.1316 | 4.2545 |
| local_mean_s43 | 4.5838 | 4.2564 | 8.9917 | 4.4063 |
| local_mean_ensemble | 4.5982 | 4.2837 | 8.9240 | 4.4223 |
| local_tight_s29 | 4.6381 | 4.4280 | 8.9129 | 4.4273 |
| local_mean_s17 | 4.6529 | 4.3302 | 9.0916 | 4.4651 |
| local_tight_ensemble | 4.6698 | 4.5064 | 8.9706 | 4.4646 |
| local_tight_s17 | 4.6804 | 4.4419 | 8.9172 | 4.5008 |
| local_mean_s29 | 4.6968 | 4.2749 | 8.9710 | 4.5460 |
| cnn_s43 | 4.7007 | 4.5379 | 8.8919 | 4.6795 |
| shared_tight_s29 | 4.7291 | 4.4028 | 9.8329 | 4.4994 |
| global_mlp_s17 | 4.7497 | 4.3450 | 10.0382 | 4.4513 |
| control_median_mlp | 4.7605 | 4.3288 | 9.5769 | 4.5118 |
| global_mlp_s29 | 4.7695 | 4.1751 | 10.6908 | 4.4036 |
| control_color_mlp | 4.7807 | 4.1292 | 10.3167 | 4.4641 |
| local_tight_s43 | 4.8109 | 4.5436 | 9.2429 | 4.5765 |
| cnn_s29 | 4.8151 | 4.2954 | 9.3407 | 4.7328 |
| control_median_poly2 | 5.0092 | 4.6852 | 9.7760 | 4.7677 |
| cnn_s17 | 5.0164 | 4.6641 | 9.8988 | 4.6545 |
| control_color_ridge | 5.2225 | 4.7188 | 10.3730 | 4.7300 |
| control_median_ridge | 5.5123 | 5.2385 | 10.1516 | 5.1947 |
| control_hist_ridge | 8.0536 | 4.8284 | 27.3041 | 6.9742 |
| control_color_hist_ridge | 8.2722 | 4.4510 | 25.7718 | 7.1413 |
| control_color_hist_mlp | 9.1778 | 7.0650 | 26.9583 | 8.4183 |
| control_hist_mlp | 10.2095 | 7.3326 | 29.9781 | 9.5529 |

## Risk-head replications and mechanism

Six subject-held-out folds produced 18 color fits. Heads used only those OOF residuals. Each arm received the same three candidate head families/budgets; 24 fits total. Primary C+ selected standard MLP, Proposed selected HGB using source validation only. Consequently the primary result compares matched search procedures, not an isolated same-head-family feature ablation. Inputs are 40 versus 49 features. The additional nine encode patch disagreement and cross-seed disagreement. Ensemble confidence and algorithm combinations are existing ideas, not established novelty.

| Version | C+ mean at 80% | Proposed mean at 80% | Difference |
|---|---:|---:|---:|
| single17 | 4.3463 | 4.2382 | -0.1081 |
| single29 | 4.0655 | 4.0396 | -0.0259 |
| single43 | 4.4517 | 4.2370 | -0.2148 |
| ensemble | 4.3328 | 4.1591 | -0.1736 |

All four patient-cluster intervals cross zero. Huber recurrence, tighter residual thresholds and removing scene context previously failed to provide reliable source gains; these negative ablations are preserved in the source reports.

## Camera and capture strata

| Primary model stratum | Images | Mean | p95 |
|---|---:|---:|---:|
| device: SLR | 152 | 3.8599 | 7.4467 |
| device: ipod | 248 | 4.8230 | 10.2868 |
| image_type: clinical: close-up | 98 | 4.5059 | 9.5339 |
| image_type: dermoscopic | 302 | 4.4411 | 9.1208 |

These device differences mix camera, person and capture effects; they are not a paired camera causal experiment. MSKCC used Canon SLR and iPod Touch with dedicated capture hardware/software and initial white balance. These are not ordinary uncalibrated iPhone/Android facial selfies. No strict unseen-camera skin result is established.

## Compute and reproducibility

Primary color model: **2,774,796 parameters**, **11,113,890 checkpoint bytes**, RTX 4060 batch-1 GPU median **1.4699 ms**, p95 **1.7267 ms**, PyTorch peak allocated **20.04 MiB**. Timing starts from prepared patch descriptors and excludes disagreement extraction, density, CPU head, JPEG and localization. The CPU Proposed head+calibrator separately takes median 0.5018 ms on prepared features. These component timings are not end-to-end latency.

Single patch model has 924,932 parameters, measured training peak allocated 108.407 MiB in the original recipe. Ensemble members train sequentially. Earlier original-JPEG preparation alone measured median 91.79 ms on 10 TRAIN images (read/hash/decode/crop/resize/all descriptors; possible OS cache). GPU allocator measurements are not whole-process VRAM. No new skin ONNX/TensorRT deployment is claimed.

Independent audit: 49 prediction arrays, 294 coverage cases, 8 exact head/calibrator replays; scalar CIEDE2000 metric maximum gap 1.42e-14. All 18 OOF checkpoints and excluded-person scalers also replayed. Full repository suite: 262 passed, 14 historical ONNX warnings, 32.57 seconds.

## Meaning for Luma and next decision

IMPLEMENTED: a compact local photograph-to-native-Lab estimator and expected-color-error/rejection subsystem. MEASURED: actual instrument-referenced skin-color error on 400 held-out-person images. NOT PROVED: accurate facial skin measurement from ordinary unseen phones, physical camera independence, instrument-level trueness, reliable cosmetic shade decisions or a defensible new neural-method advantage.

The working product aspiration remains median DeltaE00 <=2 and p95 <=5 at >=80% accepted coverage on supported independent facial-phone data. It is an engineering target, not a universal cosmetics standard. The present primary method has median3.8834/p958.0966 at80%, so it fails that target.

NEXT: retain the ordinary fusion as a frozen comparator and investigate source-only scene/subject shift and within-site capture consistency under the direct skin-color endpoint. Do not re-open this test as a tuning loop. A new confirmatory phone/facial claim needs a distinct appropriately licensed instrument-referenced cohort. Model novelty remains unverified; all angular and synthetic negatives remain archived.
