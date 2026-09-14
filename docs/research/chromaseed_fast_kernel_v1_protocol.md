# ChromaSeed-KF v1: on-demand kernels and sampled bandwidth

2026-09-13. Prospective follow-up, before any fit on real rows in this series. Previous K turn verified as **progress**: terminal process/progress, passing independent audit, source bindings and completed full replay. Broader compact/fast/high-quality model goal stays active. This experiment changes training cost and tests a modest size increase; it does not make phone-face validation available.

## Data and fixed endpoints

Only original TRAIN cache SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`:966 rows/24 people. Load color36, native Lab, patient/site/device only; no image arrays or legacy validation/calibration/test. Reuse the exact original mixed18/6, SLR→iPod8/16 and reverse16/8 person roles, inner three folds and person→site→image weighting. These roles are historically exposed and overlap. Group transfer also changes people/color distributions.

Primary endpoint: mean per-person native CIEDE2000. Secondary: image/site-person means,p90,seed variation, numeric payload, real CPU model latency, standalone fit time, synthetic scaling time/memory and complete search cost. No independent validation, patent novelty or face recognition claim.

Reuse stored-FP32 fit normalization/canonical coordinates and FP32 centers/coefficients/width with FP64 kernel arithmetic from K. Kernel `exp(-mean((x-c)^2)/(2 width^2))`, weighted Nyström RKHS ridge; landmark eigen floor1e-8 of maximum. Gaussian diagonal exactly1. Ridge alphas0.1/1/10, width factors0.5/1/2, seeds17/29/43. Nominal ranks64/128/256, using available nested centers if rank exceeds fit support or numerical rank. The256-center final payload is40,252 numeric bytes if all256 centers are available. Record realized sizes/ranks.

## Five registered training arms

1. `dense_exact`: unchanged K median/Gram/weighted randomized-Cholesky/Nyström functions. This is the new exact control, including256 centers. Reproduce old K64/K128 fit-only centers and predictions under old hyperparameters; maximum native-Lab component difference0.002, report actual drift.
2. `column_exact`: same exact all-pairs median, but evaluate each required Gaussian column on demand, retaining N×rank Cholesky factors and raw selected columns. No full N×N kernel is constructed by landmark/readout code. The exact bandwidth calculation still has quadratic allocation and must not be described as wholly linear memory.
3. `column_pairs1024`.
4. `column_pairs4096`.
5. `column_pairs16384`.

Sampled arms use on-demand columns plus the lower median positive RMS distance of a uniform random sample of distinct-row pairs, with replacement across pairs. Draw integer rows i∈[0,N), j0∈[0,N−1), map j=j0+(j0>=i). Use a two-column interleaved RNG call, so budgets are exact prefixes of the same pair stream. Pair seed is landmark seed+104729, separate from landmark RNG. Pair generation, distances and positive-distance filtering use fit rows only. Duplicated content can yield zero distances, excluded as in exact K. If the sampled stream has no positive distance, use lower median positive distance from the first fit row to all fit rows (linear fallback); if none, use1e-6. Record this fallback. No hidden quadratic fallback is allowed.

Use the same normalized features and target statistics for all arms within a fit bank. Exact width is shared over seeds; sampled width varies with the specified seed. Prefixes are nested per arm/seed/width. Cholesky pivots sample the remaining diagonal of sqrt(W)Ksqrt(W); diagonal tolerance is the unchanged1e-12×max(initial diagonal,1). On-demand column entries use FP64 norm/dot arithmetic and force the pivot diagonal to1; symmetrize the small landmark Gram before eigendecomposition. Numerical comparisons, not an assumption of bitwise equality, establish equivalence to dense code.

Collapse the whitened Nyström solve to a standard fixed kernel payload; no teacher or backpropagation. Do not store the Cholesky training factors or sampling stream in the deployed model. Final inference is unchanged `chromaseed_kernel.predict_kernel`.

## Search and policies, fixed before outer evaluation

Each bank:5 arms×3 seeds×3 widths×3 ranks×3 alphas =405 analytical readouts. Nine inner banks =3645; three final banks =1215. These4860 configurations reuse normalization, pair streams, columns and algebra; they are not4860 standalone preprocessing/training jobs. Final banks can compute the full registered grid using only fit rows, but outer metrics are computed only for selected configurations.

For every role/arm/nominal rank select width+alpha by inner OOF person error averaged across the three seeds; ties prefer smaller alpha then width. Freeze all45 hyperparameter choices before final banks/outer evaluation. There are135 selected models, each a separate seed, not an ensemble.

For each role/rank additionally select a fast arm from `column_pairs1024`,4096,16384, with `column_exact` as fallback. A sampled arm is eligible when its mean OOF person error is at most the corresponding column-exact error+0.05 and its mean imagep90 is at most exactp90+0.10. Select the smallest eligible pair budget; if none, choose exact. This is a training-policy alias of already selected models, not extra fitting. Freeze all9 choices.

One additional size-policy alias per role chooses the smallest nominal rank among these fast arms whose OOF person error is within0.05 of the best rank and whose p90 is within0.10 of that best-person-error rank (ties for best prefer smaller rank). All3 choices are inner-only and locked. This is adaptation at training/configuration time, not per-image dynamic connections or inference early exit. Report each rank as well as this policy to avoid concealing tradeoffs.

## Verification and performance

Before real fits: tests for pair-stream prefixes/no self-pairs/median/fallback, dense-vs-column Cholesky reconstruction/pivots, nonuniform weighting, Nyström collapse equivalence, degenerate cases and actual requested column count. Preserve old K sources. Synthetic prototype checks may determine implementation details before freezing, never real outer labels. Prior K CPU/GPU preflight found small compact spectral solves faster on CPU, so this series uses CPU for compact algebra; no GPU speed claim. RTX4060 remains available.

Independent audit must re-derive normalization/widths, verify fit/held person separation, selected center membership and payload/artifact hashes, recheck hyperparameter/fast-arm/size choices, and refit every final selected model from direct dense kernels/SVD with its recorded centers. Independently recompute all final predictions/metrics. Maximum native-Lab component drift against independent refit0.002; report actual values. Dense-vs-column exact controls must agree within this tolerance under matched settings. Timing output is accepted only after this audit.

Measure actual batch-one CPU inference for all135 selected models with20 warmups and3 passes over all role queries. Standalone selected fits use seed17 for every role/arm/rank: one warmup+three timings, with no cross-fit reuse. Include weighting, normalization, width computation, landmarks and solve; exclude file loading/selection/persistence. Exact and sampled methods must be timed under the same scope.

Synthetic scaling uses N=1024/4096/8192,rank128,alpha1,width factor1,seed17 for `dense_exact`,`column_exact`,`column_pairs4096`. Synthetic color36/target data are deterministic, with no image realism/accuracy claims. One warmup+three timings, incremental allocation peak measured with NumPy-aware tracemalloc per fit; label this separately from process RSS. Record actual kernel entries and retained array sizes. These fixed synthetic sizes/budgets are not selected by real outer accuracy. Verify the allocation accounting on a known NumPy allocation first. Performance on these sizes is not evidence for millions of samples.

Freeze this protocol and new core/runner plus reused source hashes before real fits. Write new run/progress files. Audit/runtime/report code may be implemented later with separate source/dependency hashes. Preserve all failed controls and prior source locks. No publications, image uploads, dataset acquisition or delegation.

Established methods: [Chen et al., Randomly pivoted Cholesky](https://arxiv.org/abs/2207.06503) describes evaluating only selected matrix columns; [Rudi et al., Nyström regularization](https://arxiv.org/abs/1507.04717) studies subsampling as computational regularization. This implementation/measurement is not a claim to invent these methods. Dataset/derived-weight rights remain separate from code rights.

Pre-freeze synthetic verification:24 relevant numerical tests passed (13 new +11 frozen K). At734 synthetic rows/rank128, one warmup+three repetitions: dense exact60.31ms, column exact55.79ms, sampled1024/4096/16384 pairs7.17/7.88/10.05ms. Dense-column query drift was0 in this synthetic probe. Kernel entries538,756 dense versus93,952 on-demand. NumPy allocation probe: known8,388,608B array, tracemalloc peak increment8,388,912B, confirming this allocator is visible. These are synthetic implementation checks, not real skin accuracy or a final production latency claim. Receipt: `experiments/runs/chromaseed_fast_kernel_preflight/runtime.json`.
