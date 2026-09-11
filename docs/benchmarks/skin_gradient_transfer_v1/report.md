# TRAIN-only skin gradient/update diagnostic

No new neural fit or held-out accuracy improvement. Six frozen929,297-parameter models; 24 partitions,48 objective cases,288 transient steps. Original TRAIN only. Original MSKCC CC-BY instrument skin references; no additional data/weights.

## Does person identity explain gradient conflict?

Numbers below describe standardized native-Lab MSE gradients, NOT color accuracy or illuminant angles. The same image-mean pooled objective is preserved by group-size weighting. Three shuffled partitions retain every group size and image. Equal-pair cosine measures gradient agreement only.

| Frozen seed17 model | True-person cosine | Shuffled cosine range | True negative pairs | Pooled color/mode cosine | Weighted auxiliary/color norm |
|---|---:|---:|---:|---:|---:|
| mixed__image__s17 | 0.0164 | 0.1411 to0.2108 | 50.72% | 0.5106 | 0.0423 |
| mixed__person_color__s17 | -0.0041 | 0.0671 to0.1261 | 48.55% | 0.0244 | 0.0696 |
| from_SLR__image__s17 | -0.1155 | -0.1290 to-0.1159 | 57.14% | 0.1074 | 0.3262 |
| from_SLR__person_color__s17 | -0.0582 | -0.0909 to-0.0704 | 64.29% | 0.7539 | 0.2441 |
| from_ipod__image__s17 | -0.0242 | -0.0546 to-0.0167 | 51.67% | 0.3082 | 0.1319 |
| from_ipod__person_color__s17 | -0.0419 | -0.0418 to-0.0066 | 55.00% | 0.4890 | 0.3310 |

Mixed-camera models show less agreement between true-person gradients than between shuffled groups. The single-camera models do not show this consistent separation. Mixed-camera shuffling also mixes cameras, capture conditions and skin-color distributions; it cannot isolate harmful person shortcuts. Legitimate label heterogeneity and cancellation near pooled stationary points remain counterexamples.

All six full-TRAIN pooled color/mode cosines are positive; weighted auxiliary norm ratios range 0.0423 to0.3310. This is not a causal intervention on the auxiliary task throughout training. It neither proves the task necessary nor supports blaming it for current generalization failure. Earlier minibatch/teacher diagnostics used different checkpoints and averaging, so their cosines are not directly comparable.

## Actual skin color under finite TRAIN interventions

Every step starts from the frozen state and is immediately undone. Step sizes are absolute parameter L2 lengths, not AdamW learning rates. Three group indices and two lengths were fixed before observations. The derivative objective is MSE or MSE+0.1CE; actual instrument-reference DeltaE00 is separately recomputed.

288/288 steps reduce their own group objective; 116/288 increase full-TRAIN mean DeltaE00. These correlated interventions are not independent samples, a significance test, or new generalization performance.

| Parameter step length | Maximum first-order remainder | Median relative remainder | Mean skin DeltaE00 change range |
|---:|---:|---:|---:|
| 0.0001 | 4.4313187e-06 | 0.005137 | -0.000656 to+0.000634 |
| 0.001 | 0.00044189211 | 0.051206 | -0.005714 to+0.007174 |

Relative remainder is max absolute group remainder divided by max first-order magnitude within that intervention. It is not relative color error. Local first-order predictions become less precise for the larger step; amax and other piecewise operations do not guarantee globally smooth behavior.

![Diagnostic controls and TRAIN color changes](gradient_transfer.png)

## Verification and compute scope

Audit PASS:6 independently recomputed pooled gradients,12 recomputed group Gram matrices using batch17 instead of32,288 intervention arithmetic checks,12 independently implemented finite-update replays, 189,336 scalar CIEDE2000 cases. Maximum scalar gap 5.77e-15; gradient/Gram gap2.49e-14; finite replay gap1.42e-14. All original checkpoint hashes remain unchanged.

The run used FP64 copies for derivative diagnosis, not deployed precision. Total diagnostic computation 57.12s; maximum allocated GPU memory 945.35MiB on RTX4060. These are diagnostic runtime/memory, not training VRAM or batch1 inference latency. No new ONNX/TensorRT/export result.

## Decision and limits for Luma

Do not infer universal harmful gradient conflict or prioritize a large gradient-invariance sweep from these snapshots. Removing auxiliary mode loss or direct squaredDeltaE00 optimization has already been tested in earlier source experiments; do not relabel those as new mechanisms.

Gradient matching is existing prior art: [Fish](https://arxiv.org/abs/2104.09937) approximates an inter-domain gradient-inner-product objective; [Fishr](https://proceedings.mlr.press/v162/rame22a.html) matches gradient variances. This diagnostic reproduces neither published method or author scores. No author code or weights were adopted.

Independent skin evidence is unchanged: primary mean4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447 DeltaE00. No strongest-baseline win, facial-phone validation or camera independence. The next decision is documented separately; the unbounded innovation goal remains active and unmet.
