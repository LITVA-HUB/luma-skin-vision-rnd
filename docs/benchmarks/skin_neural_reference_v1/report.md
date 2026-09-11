# Neural-reference skin adapters: added capacity does not give a universal gain

27 matched adapter fits and9 unchanged compact cores, on original MSKCC CC-BY real photographs with instrument-native skin Lab. Independent TEST/CAL were not opened. Source validation has been heavily reused, including historical selection of these core checkpoints. These are exploratory results; no new independent facial-phone accuracy claim.

## Measured skin color

Each entry aggregates three individual seeds, not an ensemble. SD describes seed variability only; median/p95 are means of seed-specific quantiles, not pooled quantiles or confidence intervals. Smaller DeltaE00 is better.

| Training / domain | System | Mean ± seed SD | Median | p95 | Error >10 | Mean at80% |
|---|---|---:|---:|---:|---:|---:|
| mixed /known | base | 3.4406 ±0.0540 | 2.9776 | 6.9316 | 0.38% | 3.2403 |
| mixed /known | residual | 3.6198 ±0.0207 | 3.1246 | 7.6979 | 1.26% | 3.4049 |
| mixed /known | mean | 3.5332 ±0.0549 | 3.0605 | 7.2908 | 0.51% | 3.3346 |
| mixed /known | affine | 3.5182 ±0.0533 | 3.0376 | 7.4892 | 0.51% | 3.3266 |
| from_SLR /known | base | 3.3198 ±0.0202 | 3.1374 | 6.1355 | 0.00% | 3.1820 |
| from_SLR /known | residual | 3.3596 ±0.0741 | 3.1503 | 6.5073 | 0.00% | 3.1591 |
| from_SLR /known | mean | 3.3578 ±0.0238 | 3.1987 | 6.0581 | 0.00% | 3.2178 |
| from_SLR /known | affine | 3.3037 ±0.0072 | 3.0117 | 6.2863 | 0.00% | 3.1675 |
| from_SLR /unseen | base | 5.0193 ±0.2123 | 4.8340 | 9.7825 | 4.55% | 5.0384 |
| from_SLR /unseen | residual | 5.4398 ±0.2291 | 5.1890 | 10.4599 | 7.83% | 5.5616 |
| from_SLR /unseen | mean | 5.0618 ±0.1964 | 4.8164 | 9.8460 | 5.30% | 5.0937 |
| from_SLR /unseen | affine | 4.9658 ±0.2317 | 4.7589 | 9.6942 | 4.80% | 5.0140 |
| from_ipod /known | base | 3.6138 ±0.0414 | 3.1536 | 7.2695 | 2.02% | 3.6023 |
| from_ipod /known | residual | 3.9013 ±0.0185 | 3.3312 | 8.5108 | 1.52% | 3.9031 |
| from_ipod /known | mean | 3.7195 ±0.0417 | 3.2281 | 7.6561 | 2.02% | 3.7111 |
| from_ipod /known | affine | 3.7123 ±0.0689 | 3.1770 | 7.8060 | 2.02% | 3.7196 |
| from_ipod /unseen | base | 5.9635 ±0.3915 | 5.5527 | 11.0896 | 8.59% | 6.2122 |
| from_ipod /unseen | residual | 6.4545 ±0.3539 | 6.1740 | 12.1336 | 12.12% | 7.0439 |
| from_ipod /unseen | mean | 6.0437 ±0.3787 | 5.6978 | 11.1263 | 9.60% | 6.3806 |
| from_ipod /unseen | affine | 6.4782 ±0.1610 | 5.9807 | 11.9513 | 12.88% | 6.8604 |

The local-affine mechanism beats the ordinary residual C+ head in mixed and forward transfer, but loses on reverse unseen-camera mean. Among the mixed and two unseen-camera endpoints, it improves its unchanged strong core only forward:5.0193 ->4.9658 (about1.07%). Mixed3.4406 ->3.5182 and reverse5.9635 ->6.4782 worsen. Reference mean is better than affine on reverse unseen transfer, but still loses the unchanged core. No universal capacity or mechanism win.

Previously locally reproduced stronger transfer recipes remain forward4.8328 ([paired expert](../skin_expert_anchor_v1/report.md)) and reverse4.9736 ([training branch](../skin_train_branch_v1/report.md)). These historical means use different training recipes and are contextual comparators, not new matched refits. Neither is beaten by this phase. No author-reported numbers appear as local results.

## What changed and what was held equal

All arms share a frozen929,297-parameter CaptureColor core and a193,795-parameter head:551->256->192->16->3. Total1,123,092, within the authorized1,129,297 cap. Inputs are512 neural context features,36 image descriptors and3 base color values. All heads start with exactly zero correction and identical weights; each receives the same300 AdamW steps, image draws and standardized Lab MSE supervision. Final adapters are used without validation epoch selection.

The ordinary residual head directly predicts a bounded3-vector. Reference heads learn16-dimensional affinities, retrieve residuals from TRAIN skin references and gate either their weighted mean or their locally affine prediction. The affine solver uses normalized weights and0.01 slope regularization. Every training query masks every image of its person from reference memory. Validation people are absent from core training and the bank. One-image inference uses no camera ID or query color reference.

This is an implementation of known building blocks. [MetaOptNet,CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html) and [differentiable closed-form solvers,ICLR2019](https://arxiv.org/abs/1805.08136) precede learning embeddings through regression/optimization. No claim of inventing these mechanisms, no imported third-party implementation or weights.

## Training residual diagnostic

27/27 adapters reduce full TRAIN standardized MSE relative to the frozen core, when the reference bank excludes the query person. But the core itself saw those TRAIN people. This is NOT OOF error and cannot supervise a calibrated uncertainty head. Lower training residuals together with worse source validation show that extra capacity can fit errors without improving transfer. This is consistent with residual overfitting/acquisition bias, but does not identify either as the sole cause.

The reference bank contains in-sample residuals of the core. Excluding a query person from the bank does not make that core an excluded-person encoder. The next experiment must investigate this mismatch directly before increasing the head or repeating correction passes. Full diagnostic values are in summary.json.

## Risk and coverage

All arms share nearest-bank standardized descriptor-distance ranking. The report retains360 seed-specific fixed-coverage checks and120 aggregate rows at100/95/90/80/70/60%. Curves average individual-seed risks. This common ranking isolates color changes; it is not an error-calibrated C+ confidence head or a refusal guarantee.

![Full risk-coverage curves, displayed from60%](risk_coverage.png)

## Measured compute and payload

CUDA measurements on RTX4060. Cached-head training includes the frozen core and source tensors resident in the process. Batch1 latency includes neural core and reference correction on already prepared GPU tokens/descriptors; JPEG decoding and image-feature extraction are excluded. Process VRAM includes training caches and copies, so it is not isolated production VRAM.

| Adapter | Parameters | Head train peak MiB | Batch1 median ms range | Complete checkpoint bytes | Extra reference scalars |
|---|---:|---:|---:|---:|---:|
| residual | 1,123,092 | 75.181–79.566 | 0.725–1.119 | 4505423–4505423 | 0–0 |
| mean | 1,123,092 | 81.353–100.932 | 0.978–2.283 | 4529999–4578895 | 6137–18354 |
| affine | 1,123,092 | 84.767–110.073 | 1.424–3.042 | 4529999–4578895 | 6137–18354 |

Reference memory stores16 embedding plus3 residual scalars per TRAIN photograph; it is included in checkpoint bytes and separately counted. Fixed preprocessing/target scales add1102 scalars. Private original features/identifiers are not part of the deployed checkpoint. No ONNX/TensorRT export or end-to-end phone latency claim.

## Verification and Luma boundary

All9 source feature/core replays match.27 NumPy head replays include2376 independent augmented least-squares solves;36 excluded-person perturbations pass. Three complete affine optimization refits reproduce exact final head hashes.60 deployment prediction arrays reproduce exactly;9504 independent scalar CIEDE2000 cases and360 coverage rows pass. Max NumPy native Lab discrepancy4.77e-6; scalar color discrepancy5.78e-15.

The larger model and reference correction are implemented and measured on genuine instrument-referenced skin color. They have not established novel-model superiority or ordinary smartphone facial accuracy. Independent evidence remains primary mean4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447. The desired median<=2 and p95<=5 at>=80% on unseen ordinary phones remains unmet.

[Frozen protocol](../../research/skin_neural_reference_protocol_v1.md) · [Next decision](../../research/skin_neural_reference_next_decision.md)
