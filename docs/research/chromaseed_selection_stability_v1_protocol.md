# ChromaSeed-S: fixed-OOF selection sensitivity

Registered 2026-09-13, before calculation. User-authorized autonomous compact-model research. This is a diagnostic of the completed W search, not a new model-selection rule or an accuracy trial. No delegation, new data acquisition, deployment or publication.

## Question and boundary

W's expanded grid reduced all 15 inner scores while worsening 13 matching-family outer scores. Those outcomes motivated this diagnostic; they cannot be made independent by resampling. Determine how much W's inner choices depend on fit-side people. Do not use the exposed outer predictions/results to choose a rule, threshold, model or winner. Preserve all W/P/KE and earlier code/data/artifact locks.

Read only original TRAIN `train.npz`, SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`. Load only `target`, `patient`, `device`. No color/RGB/tokens, old validation/calibration/test or outer model/prediction archives. Read W's nine inner OOF archives/receipts, W/P selections and source manifests; W's verification is a provenance binding. No new fits and no GPU timing.

Use the frozen three roles and three inner person-disjoint folds. Fit-side people: mixed 18 (6 SLR, 12 iPod), SLR-only 8, iPod-only 16. Roles overlap and are never independent replications. Every prediction must have been produced in the person's held-out fold. All selected scores and all 891 family-candidate scores must reproduce W to absolute tolerance 1e-10 before diagnosis.

## Registered calculation

For each of five families in each role, take all W candidates including zero-step aliases. Compute native CIEDE2000 per image for each of seeds 17/29/43 separately; average errors within each person, then across seeds. The person, not the image or seed, is the sampling unit. This is not a prediction ensemble. No site reweighting: match W's primary person mean exactly.

Preserve the original tie order: lower mean error, then fewer steps, lower alpha, narrower width. Pre-sort by the last three fields and use first minimum. Do not introduce a near-tie tolerance to change choices. Report exact configuration switches separately from their full-OOF score costs, since different settings can have similar scores or identical stopped-trajectory predictions.

1. Remove each fit-side person from the selection score once, keeping saved predictions fixed. Recalculate the winner and the original winner's rank. Report switch count, configuration frequencies and full-OOF regret of new choices (their full-OOF mean minus the original winner's full-OOF mean). This is score-deletion sensitivity, not leave-one-person-out model retraining or a fresh held-person accuracy estimate.
2. Draw 20,000 person bootstrap samples per role with replacement, separately within each camera, preserving its original person count. RNG seeds are 2026091301, 2026091302, 2026091303 in mixed/SLR-only/iPod-only order. Reuse each role's draws across all families/candidates. Record original-winner selection frequency/rank quantiles, all configuration frequencies, fraction with alpha below the old 0.1 boundary, fraction with positive correction steps for iterative families, and full-OOF regret quantiles.
3. Report paired resampling distributions of two fixed contrasts: original W winner minus matching-family P winner, and original W winner minus the full-OOF runner-up. The P configuration is evaluated using the exact W control OOF payload at the corresponding alpha value. Report mean, 2.5/97.5 percentiles and fraction below zero. These are conditional sensitivity ranges of already-selected settings, not corrected population confidence intervals or p-values.

No additional cutoff, selection policy or model promotion is authorized by a favorable range. In particular, frequent weak-alpha selection would not prove camera invariance; instability would not prove that it caused the outer regressions.

## Verification and artifacts

Before real calculation: synthetic numerical tests for unequal image counts, nonlinear error versus prediction averaging, exact tie order, cluster-stratified draws, hand-computable deletion/paired gaps, invalid inputs. Freeze core/runner/tests/protocol and inherited helpers/data/OOF/selection hashes.

Save only ordinal person axes in local score archives; never direct identifiers or participant images. Independently reconstruct every candidate/person score from saved OOF, recompute every choice, all deletion winners and all 20,000 resample winners using explicit loops/weighted sums separate from the primary diagnostic function. Verify source/output hashes and counters. Report all 15 groups, including unhelpful results, with one diagnostic chart and reproducible commands. Source manifests and numerical artifacts remain immutable after verification.

## Interpretation

Fitted OOF models share training people. The resampling leaves those fits and folds fixed, misses fitting/fold-generation uncertainty, and preserves camera proportions; it cannot assess a new camera or a new population. Reused TRAIN has only 24 people in total. A small or large conditional range cannot establish ordinary-phone facial quality.

Model-selection overfitting is an established risk, rather than a new claimed discovery: [Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html). Correlation in cross-validation also prevents a universal unbiased variance estimator: [Bengio and Grandvalet, 2004](https://www.jmlr.org/papers/v5/grandvalet04a.html). These motivate cautious diagnostic interpretation; they do not prove the cause of Luma's observed error.
