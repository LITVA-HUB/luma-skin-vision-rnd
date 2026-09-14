# ChromaSeed-KE v1: exact condensed-distance follow-up

2026-09-13. Registered after the KF primary audit exposed a failure of sampled-width/size selection, before any real fit of this follow-up. This is a runtime-preserving implementation study, not a new accuracy discovery or independent test. The broad user goal remains active. Preserve KF/K/R/palette sources and all negative results.

## Hypothesis and fixed algorithm

Replace only the exact all-pairs bandwidth implementation: SciPy `pdist` with built-in `sqeuclidean` returns the condensed upper-triangle distances. Divide by36 and take square root in place, exclude values≤1e-10, select the lower median in place, minimum1e-6. All-constant/single-row cases return1e-6. No pair subsampling, no approximate quantile, no new hyperparameter. This still has O(N²) condensed-distance storage; it is not a linear-memory algorithm.

Use the existing frozen fit normalization/canonical FP32 coordinates, on-demand weighted randomized Cholesky, Nyström solve, FP32 payload and FP64 inference unchanged. This combines exact bandwidth with the already audited column implementation. The standalone fixed model architecture stays64/128/256 centers and color36→native Lab.

## Data and no-selection controls

Only original TRAIN cache hash `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`, color/target/patient/site/device, no images or legacy validation/calibration/test. Use the same previously exposed roles/folds. No new dataset or external model weights.

Reuse every KF `column_exact` role/rank selected width/alpha and the same seeds17/29/43. Do not select new settings. First refit81 inner models (9 folds×3 ranks×3 seeds) under those fixed settings and compare their fit normalization, rounded widths, centers and OOF predictions to their exact KF bank entries. Then refit27 final models and compare all held-row predictions with the already audited exact KF artifacts. No coefficients or thresholds are tuned using these comparisons.

Record exact equality where it occurs and maximum differences where it does not. Require relative unrounded bandwidth difference≤1e-12, rounded width difference≤1 ULP and maximum native-Lab component prediction difference≤0.002 for every model; exceeding any limit means the replacement is not accepted. This is a numerical equivalence guard, not a bound on true skin-color error. All final metrics are recomputed, and source/payload/data hashes must verify.

## Measurement and provenance

For each role/rank at seed17, time the three fixed implementations `dense_exact`, original `column_exact`, new `condensed_exact`, with one warmup+three independent fits each. Include balancing, normalization, exact bandwidth, pivots and solve; exclude loading/selection/persistence. No cached width or matrix between fits. Parent KF already measured the same predictor; additionally check and time batch-one inference for the27 new final models (20 warmups,3 full query passes).

Synthetic scaling: the same generated8192-row prefix data/weights from the KF runtime script, sizes1024/4096/8192,rank128,seed17,width factor1,alpha1. Measure condensed-exact1 warmup+3 timings plus1 separate allocation-trace fit per size. Compare with completed KF controls and state that they were collected separately on the same host. Peak NumPy-aware tracemalloc means incremental traced allocations, not process RSS or deployment RAM. The exact method still scales quadratically in the number of pairs.

Tests before real fits: exact lower median against frozen blocked implementation with random rows, duplicate content, large offsets, constants and single-row cases; unseen-query equivalence of the complete condensed/column-exact pipeline. Source-lock this protocol, new core/runner, parent sources/choices and SciPy version before real fits. No new hyperparameter search is performed.

SciPy1.18.1 is already installed locally; no installation is required. This standard optimized distance implementation is not algorithmic novelty. [Official SciPy pdist documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) documents the condensed form, squared Euclidean metric and built-in compiled kernels. Existing dataset/derived-artifact obligations and code/dependency licenses remain separate.
