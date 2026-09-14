# ChromaSeed-D: camera-group and target-support diagnosis

Registered 2026-09-13 before numeric inspection/calculation. Previous S turn verified as progress; all its source/input/artifact hashes match and its process handles are terminal. This diagnostic addresses the active compact/fast/high-quality goal without promoting a model from exposed outer errors. No delegation, publication or data acquisition.

## Data and interpretation boundary

Original TRAIN only: `train.npz`, SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`, 966 records, 24 people (8 SLR, 16 iPod). Load `color`, `target`, `patient`, `site`, `device`; no images/RGB/tokens or old validation/calibration/test. Do not read prior outer prediction/results archives. Previous S verification/manifest is provenance only.

`color36` contains 27 RGB quantiles, mean RGB at positions27:30, population std3 and correlations3. Native Lab targets are instrument-referenced. Camera labels are acquisition metadata, not an inferred demographic label. People and acquisition are confounded with camera. No classifier score identifies a causal camera effect, and all TRAIN roles are historically reused.

## A. Fixed camera classifiers

Use frozen `folds_for` on the whole original TRAIN, yielding three camera-stratified person-disjoint folds; every person is predicted once. No hyperparameter search. Predict +1 for SLR, -1 for iPod. Fit weights allocate half the mass to each camera, equal people within camera, equal sites within person, equal images within site; sum1.

Four input views: color36, mean RGB3, native Lab3, color36 residual after a fit-only linear regression on native Lab. The last two use an oracle target and are diagnostic controls, unavailable to a deployed selfie model. Residualization: weighted standardization of color36 and Lab on fit rows (population variance, std floor1e-6), weighted least squares of standardized color on intercept+standardized Lab, `lstsq(rcond=1e-12)`, then fit-only standardization of residuals. This removes a linear target relationship only, not all skin-color information.

Fit two fixed classifiers for each view and fold (24 fits):

- Linear ridge with unpenalized intercept and alpha0.1, minimizing sum(weight*(score-label)^2)+alpha*sum(nonintercept coefficients^2).
- Gaussian kernel ridge, no intercept, alpha0.01, using weighted kernel system `sqrt(W) K sqrt(W)+alpha I`. Width is the positive lower median of fit-only standardized RMS pair distances (distance>1e-10, floor1e-6); kernel exp(-RMS_distance_squared/(2*width^2)). No approximate/sampled width.

Evaluate each person's score as the equal-site mean of image scores. Report AUC (SLR positive, exact ties half), balanced accuracy at score>=0 and correct-person counts in each camera. Also report each fold. No confidence intervals, p-values or threshold tuning; correlated OOF fits and only24 people do not support broad confirmation. Lack of predictability by these two classifiers would not prove identical distributions.

## B. Held-fold target-color matching

Compute each person's native-Lab centroid as the equal-site average of site means. Within each held fold, pair SLR and iPod people by centroid DeltaE00. For each fixed caliper1,2,3,5,10 independently, choose maximum cardinality matching, then minimum total distance, with at most one use per person. Implement assignment with sufficiently expensive dummy/unmatched slots; verify against exhaustive matching enumeration in these small held folds.

Report pair count, matched/unmatched people by camera, total matched images and pair-distance summaries at every caliper. Do not choose a caliper from its performance. Using the already fixed OOF scores, report AUC/balanced accuracy on the paired people and fraction of within-pair SLR scores above iPod (ties half). A caliper with no pairs gets null metrics. Matching uses query targets solely for retrospective conditioning; no classifier, normalizer or threshold is fitted on them. Centroid matching does not equalize within-person color/site distribution, acquisition or other person differences. It is not paired acquisition of the same person.

## C. Coverage of the three historical roles

For mixed, SLR-to-iPod and iPod-to-SLR roles, separately compute nearest-fit-row distances in weighted-standardized color36 RMS and native Lab DeltaE00. All color standardizers fit on fit-side rows only, with equal person/site/image weights summing1. Reference distances for fit rows exclude every row of their own person. Query distances use only fit rows.

Set a descriptive support radius to the weighted empirical 95th percentile of those fit reference distances: smallest sorted value whose cumulative weight reaches0.95. Report fit reference and query weighted means/medians/p95 and query mass strictly above the radius. This is a geometric diagnostic, not calibrated out-of-distribution detection or an error guarantee. Do not select a model using these values. Report native-Lab 5/50/95 percentiles per camera and equal-site person centroid spread without exporting identifiers.

## Verification and frozen artifacts

Tests before real calculation: camera/person/site weights, fit-only normalization, ridge equivalence to independent augmented least squares, kernel solve equivalence, exact tie AUC, target residual control, caliper maximum-cardinality behavior, own-person exclusion, empirical weighted quantile. Freeze source, protocol, tests, environment and parent verification before calculation. Preserve all earlier primary locks and dependencies.

Save local ordinal-axis scores/pairs/support distances; never direct IDs or participant images. Independently recompute all 24 fit-only pipelines and predictions via separate normalizers and SVD solvers, all person/fold/matched metrics, exhaustive matching objectives, all six support calculations and target summaries. Report all views/calipers/roles, numerical drift, process status, reproducibility commands and a next decision. No new skin-color accuracy/model-size/latency claim follows from a diagnostic classifier.

Related work uses classifiers to study distribution differences: [Jang et al., ICML2022](https://proceedings.mlr.press/v162/jang22a.html). We do not implement their sequential test or assume pure covariate shift. Assignment and AUC references: [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html), [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). No new packages, code or data downloaded.
