# CC v2 protocol and interpretation

This is an improvement experiment after frozen public feasibility milestone7637d6d. The synthetic negative result2685bf0 and v1 positive/negative measurements remain intact. Source-selection records are timestamped JSON beside this file. Target IDs were selected independently before pixel/GT inspection; source estimator selection precedes target decoding, selector fitting precedes target error evaluation.

## Source-only choices

SimpleCube++ official train population is partitioned by capture date into1126 estimator-training,119 model-validation,259 risk-fitting and268 calibration images. Official462test images have already been observed in v1 and66of67test dates overlap development dates. Therefore that set is a regression diagnostic; it cannot be represented as newly independent confirmation.

Screen MobileNetV3Small direct/GW-anchored/SoG-anchored modes, each with diagonal augmentation0/0.7, seed17,120epochs. Then test the three matched larger backbones with gain0; small anchored variants lose source validation, making a capacity diagnostic useful. All choices use unaugmented source-validation mean reproduction error, never target error. Larger backbones and three final seeds were anticipated in the initial plan. A conventional statistics-regression source screen provides an additional cheap learned baseline. These extra source choices create model-selection multiplicity; the new camera data remain untouched during them.

The anchor-relative normalization/restoration is known prior art, including Cotogni/Cusano2022 Eq19 and2024 illuminant-equivariant networks. Exact equivariance is qualified to nondegenerate positive diagonal gains on the fixed preprocessed input. Real sensor changes are not generally diagonal. Gain-stressed real inputs do not create additional physically measured examples.

## Selective systems and matched controls

For each frozen estimator, fit conventional context-only, cheap relative-statistics-only, and combined error predictors. Ridge regularization1/10/100 and histogram gradient boosting3/7leaves receive the same five capture-date-group folds of the259 risk samples. Target is log1p reproduction angular error; choose lowest pooled out-of-fold error at80% coverage, with AURC as tie breaker. Each candidate's scores and selection evidence are retained. Refit on all risk groups. Calibrate with a positive scalar matching mean predicted/observed reproduction error on the separate268calibration images (scale floor1e-8 preserves ranking in degenerate perfect-calibration cases). This is ordinary post-hoc calibration, not a distribution-free safety guarantee.

C+ is the corresponding ordinary compact learned estimator with a standard confidence/error head and calibration. The same estimator plus combined relative features is an additional strong control: a proposed normalization mechanism must be compared with this too. Existing v1 checkpoints and original selectors are also preserved; source-only nonlinear selector upgrades are evaluated separately and identified as such. Report every failed architecture/selector; no publication claim from best-of-test selection.

Metrics: angular recovery and reproduction error in degrees; mean,median,trimean,best/worst25%,p90,p95,max, fractions>10°/>20°. Risk-coverage curves rank predictions by estimated error with deterministic image-ID hash ties. Fixed100/95/90/80/70/60% points use floor(N*coverage), so achieved empirical coverage may differ slightly. Also report frozen source-calibration thresholds, actual target coverage and actual accepted error; nominal80% is not a promise of80% target acceptance. Invalid channel-degenerate inputs are mandatorily excluded; unattainable coverage points are explicit.

## Fresh camera test

Train/select/calibrate solely on SimpleCube++ Canon550D/600D images. Evaluate128 prespecified camera-unique INTEL-TAU field images each from Canon5DSR, NikonD810 and SonyIMX135_BLCCSC. No target-camera identity, CCM, target-dataset aggregate statistics, target labels, multi-image calibration or parameter adaptation enters inference. This is a custom384-image transfer test on three cameras absent from fitting, not the authors' full INTEL-TAU10-fold benchmark. Sony001–030 and all shared-camera/lab categories were excluded before sampling.

Processed TIFFs are already black-corrected and saturation normalized; no gamma decode, repeated black subtraction or GT-chart masking is applied. TIFF BGR decoding is reordered toRGB; max>=0.98 pixels are removed. Neural input uses the same per-image thumbnail/exposure transform as v1,128x128. Classical estimators use the full processed image. The original paper says GT acquisition images are not published and CCMs are approximate illustration matrices, not reference colorimetry. [Original paper](https://pure.au.dk/ws/files/301638552/INTEL_TAU_A_Color_Constancy_Dataset.pdf).

Original license isCCBY-SA4.0; the mirror'sMITbadge is not used. Files match frozen mirror ZIP CRCs and local SHA256s; sparse extraction cannot establish identity to original multipart archive SHA256s. This provenance limit is part of the result. New data are evaluation-only; no resulting weights are trained on them.

No physical surfaceΔE00 or facial skin accuracy can be inferred from these illuminant-only labels. Improved benchmark normalization/reliability is component evidence for Luma, with facial colorimetric validation still future work.

SonyIMX135 was already observed in the V1 pilot; fresh128 Sony images are disjoint, not a previously unknown camera model to the project. Canon5DSR and NikonD810 are new to development. All three cameras are absent from estimator, selector and calibration fitting.
