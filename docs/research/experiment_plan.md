# Experiment plan

Status: **PRE-REGISTERED FRAMEWORK; REAL RESULTS NOT MEASURED.**

## Data readiness and freezing

1. Collect the pilot under `docs/data/dataset_protocol.md` and validate every `Record` plus auxiliary registration/calibration logs.
2. Compute instrument repeatability before any learned target experiment. If G1 fails, stop model training, diagnose site registration/operator/instrument effects, and repeat the protocol pilot.
3. Freeze dataset version/hash, subject allocation, exclusion rules, primary metrics, 80% primary coverage, uncertainty method, and locked unseen device/light evaluations.
4. Keep all subjects disjoint across train, validation, calibration, and test. Train fits weights/transforms; validation selects models; calibration fits the error mapping and selective threshold; test is evaluated once. Error heads use only subject-held-out or cross-fitted residuals.

## Methods

- **A0:** uncorrected robust cheek color from the original image.
- **A1:** fixed classical color-constancy candidates, evaluated individually without selecting by test performance.
- **A2:** global/device-conditioned color correction fitted only from allowed paired training/calibration data; document applicability to unseen devices.
- **C:** standard compact image/ROI regressor with ordinary anatomical cheek regions and no proposed reliability mechanism.
- **C+:** strongest fair learned comparator: same backbone, parameter budget, inputs, augmentation, optimizer/search budget, target data, and calibration access as Proposed; include standard quality/confidence and an ordinary uncertainty baseline.
- **Proposed:** C+ plus measurement-suitability ROI estimates, bounded photometric hypotheses, ambiguity features, and dedicated residual/error prediction.

All learned methods receive equal opportunity. Tune C+ seriously; a weak comparator does not support the claim. Record parameters, FLOPs where available, training time, seeds, hardware, dependency versions, code commit, configuration, and data/split hashes.

## Experiment sequence

1. **Repeatability:** within-site repeat Lab and ΔE00; operator/session effects; registration audit. Decide G1 only against the frozen tolerance.
2. **A0 instability (RQ1/G2):** within-subject camera/light/repeat variation and error against instrument targets.
3. **A1/A2 (RQ2):** paired deterministic comparisons, including failure modes and domain breakdowns.
4. **C and C+ (RQ3):** compact learned baselines at full coverage and with their strongest valid selection scores.
5. **Proposed mechanism (RQ4–RQ6):** train ROI reliability only from defensible real supervision or cross-fitted measurement residuals; train the error model from subject-held-out predictions. A generic synthetic mask is not a successful learned ROI result.
6. **Selective evaluation (RQ7):** use calibration-selected thresholds; report risk at 80% coverage and the full 0–100% risk–coverage curve with achieved coverage and abstention reasons.
7. **Domain evaluation (RQ8):** apply the frozen pipeline and thresholds to the locked unseen device and unseen lighting sets.
8. **Compression/export (RQ9–RQ10/G6):** only after positive real evidence, test a roughly <10M-parameter candidate, ONNX, FP16, and optionally INT8 against the same samples and policy.

## Metrics and statistical analysis

Primary loss is region-level CIEDE2000 ΔE00 to the registered D65/2° instrument mean. Also report component errors in L*, a*, b*, median/mean/90th and 95th percentiles, failure rate, achieved coverage, risk–coverage curves, area under the risk–coverage curve, predicted-error calibration, and probability calibration when a tolerance probability is produced.

Compute Proposed-minus-comparator effects on identical samples. Use subject-cluster paired bootstrap intervals, retaining all observations of each resampled subject. Report per-camera, per-light, cheek, and intersectional technical strata only when sample counts support them; suppress unstable summaries transparently rather than merging test data into development.

## Required ablations

- Remove measurement-suitability ROI and use the C+ anatomical ROI.
- Replace learned suitability with generic skin segmentation or fixed cheek masks.
- Remove each photometric hypothesis and evaluate original-only and best-single-hypothesis variants.
- Remove hypothesis-disagreement features.
- Remove left/right disagreement, ROI perturbation sensitivity, gradient, clipping, highlight, shadow, and domain features by logical groups.
- Replace the dedicated error head with each conventional score used by C+.
- Train the error head on proper cross-fitted residuals versus the deliberately invalid in-sample-residual diagnostic, clearly labeled to expose optimism.
- Remove device metadata and test device-conditioned correction separately.
- Compare compact student/direct model to the uncompressed reference and evaluate each export/precision step.

## Repetition and decision rules

Use multiple predeclared training seeds for learned methods and pair seeds/configurations across C+, Proposed, and ablations. Select one model-selection rule before test evaluation. Report every planned run, including crashes and excluded runs with reasons. Do not pick the best seed as the headline result.

Pass G4 only if the pre-registered primary effect favors Proposed with an uncertainty interval and magnitude judged meaningful before test access, while the full curve and major domains show no disqualifying regression. Pass G5 only if equal-coverage selection remains useful on locked conditions. Otherwise record the failure and choose the smallest evidence-driven pivot.

## Synthetic engineering track

Synthetic fixtures may run schema validation, baselines, training, calibration, risk–coverage computation, export, and API checks. Label every artifact `SYNTHETIC`. Synthetic results do not answer RQ1–RQ10, pass G1–G6, validate ROI reliability, or establish a learned scientific result.

