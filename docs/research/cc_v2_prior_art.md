# CC v2: diagonal-equivariant residual correction — bounded prior art

Evidence cutoff: **2026-09-10**. **LITERATURE REVIEW and PLANNED EXPERIMENT; NO REPRODUCED RESULTS.** This note supplements `public_color_prior_art.md`. Primary publisher/author sources were inspected; no code, weights or datasets were downloaded or implemented. This is not an exhaustive novelty search or commercial clearance opinion.

## Finding and appropriate claim

The proposed cheap-anchor normalization, unrestricted residual CNN, and illuminant restoration are a specialization of a published equivariance construction. Treat CC v2 as an empirical test of known structural constraints under a stricter source-only, camera-blind and compact-compute contract. Neither diagonal equivariance, residual correction, normalized context nor learned reliability is by itself a new contribution. The open question is measured accuracy and selective-risk behavior on genuinely held-out cameras.

## Closest primary sources

**Cotogni and Cusano, Neurocomputing 502 (2022), 110–119, DOI [10.1016/j.neucom.2022.06.118](https://doi.org/10.1016/j.neucom.2022.06.118).** The [author preprint](https://arxiv.org/pdf/2207.00292), Section 2.4, Eq. (19), already gives

```
h(x) = f(x − Gm φ1(x)) + Gn φ2(x),
φj(x + Gm Δ) = φj(x) + Δ.
```

Here `f` can be arbitrary. Section 3 turns positive diagonal RGB gains into channel offsets using logarithms. The paper also develops constrained internal layers and applies them to Color Cerberus illuminant estimation. Its NUS experiment uses whole-dataset three-fold cross-validation and artificial illuminant distortions, rather than the proposed held-out-camera contract. It floors RGB before logarithms and discusses padding restrictions for internal equivariant convolutions. **Our inference:** the proposed normalize/predict/restore wrapper falls directly within Eq. (19); choosing arithmetic or Minkowski anchors changes the pooling function, not that construction. [Full preprint](https://arxiv.org/pdf/2207.00292).

**Cotogni and Cusano, CCIW 2024, pp. 249–260, DOI [10.1007/978-3-031-72845-7_18](https://link.springer.com/chapter/10.1007/978-3-031-72845-7_18).** The publisher records first online publication on **18 October 2024**; the [institutional record](https://iris.unipv.it/handle/11571/1514055?mode=complete) labels the record 2025. Its abstract directly studies illuminant-equivariant versions of established illuminant-estimation networks on NUS, including changed illumination without requiring training augmentation. This is direct task-level overlap. Publisher abstract/metadata and institutional record were inspected, but full chapter equations and experimental tables were inaccessible in this review. Do not label a v2 implementation a reproduction of that chapter or infer additional architecture details from its abstract.

**Barron and Tsai, FFCC, CVPR 2017.** [Original paper](https://arxiv.org/pdf/1611.07596) formulates illuminant estimation as localization in log-chroma histograms, uses Fourier convolution on a torus, and produces an illuminant posterior. It is a strong compact learned baseline with a useful uncertainty output. However, Section 5/Eq. (20) introduces learned absolute illuminant gain and bias maps; gray-light de-aliasing also uses an absolute illuminant range. Thus the complete published model is not automatically an exact continuous diagonal-equivariant estimator. Histogram quantization and de-aliasing need auditing even in a no-prior variant. A covariance or posterior concentration is a candidate score, not evidence of calibrated selective risk. Use the [original Google MATLAB repository](https://github.com/google/ffcc), not an unaudited third-party port, as the reproduction reference.

## Exact mathematical contract for v2

The following is an explicit specialization/derivation for this experiment, not a new theorem. Let `I[p,c] ≥ 0` be linear RGB and `D = diag(dR,dG,dB)` with each `dc > 0`, acting identically at every pixel. Let a fixed nonempty valid-pixel mask be `M`. Require a strictly positive anchor satisfying **componentwise diagonal equivariance**:

```
a(DI) = D a(I).
```

Scalar positive homogeneity `a(tI)=t a(I)` alone is insufficient. Per-channel Minkowski means satisfy the stronger condition:

```
a_c(I) = (sum_{j∈M} I[j,c]^q / |M|)^(1/q),   q > 0.
```

Here `q=1` is Gray World; larger finite exponents give Shades of Gray, whose Minkowski connection is established in [Finlayson and Trezzi (2004)](https://research-portal.uea.ac.uk/en/publications/shades-of-gray-and-colour-constancy/). The exponent must be fixed in advance or chosen using invariant information and source validation.

Define the canonical image `N(I)=I/a(I)` channelwise and any deterministic residual predictor `rθ(N)∈R³`. Then

```
N(DI) = DI / (D a(I)) = N(I),
e_tilde(I) = a(I) ⊙ exp(rθ(N(I))),
e_tilde(DI) = D e_tilde(I).
```

Centering `r` across channels removes its irrelevant common log scale. If the final prediction is unit length, the exact statement is projective:

```
e_hat(DI) = normalize(D e_hat(I)).
```

The CNN can use ordinary padding, biases, activations and normalization **after an exactly invariant input**; special equivariant internal layers are unnecessary for this wrapper. With `x=log I`, the pooling `φ_c(x)=log a_c(exp x)` obeys `φ(x+Δ)=φ(x)+Δ`, making the relation to the 2022 construction explicit. Logarithms of zero require a documented policy.

### Conditions that must be tested rather than assumed

- Compute the quotient with the **unnormalized** anchor. Dividing by a unit-length anchor leaves a gain-dependent common intensity factor. An additional invariant exposure normalization can remove it, but must be specified and checked.
- Fixed absolute epsilon additions/floors before the anchor or quotient generally break the identity near zero. Floors or clipping applied only to the already invariant quotient preserve invariance. Empty masks or a zero anchor channel are outside the proof; flag these cases rather than silently claiming exactness.
- Recomputed brightness/saturation masks, thresholded pixel selection, RGB-dependent resize policies, sensor clipping, quantization, nonlinear rendering and additive black-level errors can violate the assumed transformation. Fixed spatial masks and fixed linear resizing commute with global gains. Audit the complete preprocessing path.
- Camera changes include spectral-sensitivity differences and generally cannot be represented by a single positive diagonal matrix for all surfaces. The proof supplies gain equivariance, **not camera invariance**. Real mixed illumination, noise and nonlinear ISP effects also remain empirical challenges.
- Removing absolute illuminant coordinates removes potentially useful camera/illuminant priors. This is a deliberate bias/variance tradeoff; lower source or target accuracy is possible even if the identity holds exactly.

## Reliability head: compatible target, unresolved calibration

Let `s(I)=h(N(I), z(I))`, where every auxiliary feature `z` is invariant. Suitable candidate features include centered log-ratios of equivariant candidates to the anchor:

```
z_k = center(log(e_k / a)),
center(v) = v − mean_channels(v).
```

Centering also removes arbitrary scalar normalization of each candidate. Raw candidate chromaticities, anchor chromaticity, absolute log RGB and ordinary recovery-angle disagreement are not generally invariant. Residual magnitude/disagreement need validation as risk predictors; neither is necessarily the actual error.

For reference `g` and prediction `e`, transformed together with the image, the corrected-neutral vector obeys `(Dg)/(De)=g/e`. Consequently **reproduction angular error** is invariant. **Recovery angular error is not:** a positive diagonal map generally changes the angle between two different RGB vectors. Use the existing project's `g/e` reproduction convention consistently; retain recovery error as a separate reported metric. See the primary [reproduction-error definition](https://www.bmva-archive.org.uk/bmvc/2014/files/paper047.pdf).

An invariant head is well aligned with reproduction-risk prediction under ideal diagonal gains. It cannot generally predict gain-dependent recovery error exactly from identical normalized features. More importantly, invariance does not imply uncertainty calibration or reliable abstention on a new camera. Train risk heads on subject/scene-held-out source residuals; fit selection thresholds on separate source calibration data. Never fit them on target labels, target batches, or the final test set.

## Recent work that narrows the experiment

| Primary work | Relevant overlap and decision |
|---|---|
| [Integral FFCC, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wei_Integral_Fast_Fourier_Color_Constancy_CVPR_2025_paper.html); [author preprint](https://arxiv.org/pdf/2502.03494) | Extends FFCC to local/multiple illuminants using integral histograms. Cheap thumbnail inference remains active prior art. Appropriate future mixed-light reference; it should not displace a global FFCC baseline for the present single-target benchmark. Official code/license not established here. |
| [Buzzelli and Bianco, Pattern Recognition 2025, 111175](https://www.sciencedirect.com/science/article/pii/S0031320324009269) | Formalizes three uncertainty forms, across five algorithm categories, using at most one inference, and uses uncertainty for cascading. Single-pass reliability is established. Full equations were not accessible; do not invent a supposedly faithful estimator from the abstract. |
| [CCMNet, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_CCMNet_Leveraging_Calibrated_Color_Correction_Matrices_for_Cross-Camera_Color_Constancy_ICCV_2025_paper.html) | Compact cross-camera estimation conditions on calibrated CCM information and augments imaginary cameras. Strong relaxed-contract reference; CCM input is outside camera-blind v2. The [supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Kim_CCMNet_Leveraging_Calibrated_ICCV_2025_supplemental.pdf) explicitly times a C5-based FFCC-operations surrogate, not the original MATLAB FFCC: preserve that distinction. |
| [BRE, Pattern Recognition 2026, 112153](https://www.sciencedirect.com/science/article/pii/S0031320325008131) | Brightness robustness training is prior art; include a matched augmentation control. Do not claim scalar exposure robustness alone as a contribution. Journal/preprint publication and result differences are recorded in `public_color_prior_art.md`. |
| [VLM-CC, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_White-Balance_First_Adjust_Later_Cross-Camera_Color_Constancy_via_Vision-Language_Evaluation_CVPR_2026_paper.html) | White-balances using a current estimate, evaluates residual cast in pseudo-sRGB and iteratively corrects it. Direct overlap with contextual residual correction. Its VLM inference and rendering contract differ from the compact camera-blind CNN experiment. |
| [CSNet, CVIU 2026](https://www.sciencedirect.com/science/article/pii/S1077314226000056) | Decomposes input into mean, variation magnitude and direction, then uses content/structure weighting for illumination prediction. Normalized/decomposed scene cues are established. Only publisher abstract/introduction inspected; compute and code rights are unverified. |
| [Saied and Fleuret, arXiv:2605.08193v3, 2026 preprint](https://arxiv.org/abs/2605.08193v3) | Wraps arbitrary backbones with normalization/denormalization for affine normalization equivariance in denoising. Different transformation group and task, but additional contemporary structural overlap with using an unrestricted network inside an equivariance wrapper. It is a preprint, not a reproduced color-constancy baseline. |

## Minimum useful comparisons and decision rule

1. **Cheap anchor alone**, plus the best source-selected existing classical estimator. Include a fixed learned residual vector to test whether a scene CNN adds value beyond a global source correction.
2. **Matched raw-input compact CNN**, with identical budget and source augmentation, and its normalized-anchor residual version. This separates the structural constraint from capacity, optimizer and augmentation changes.
3. **Full FFCC retrained on source data** without target camera fitting. If a port is needed, label it and verify numerical parity before calling it reproduced FFCC. Separately label a no-absolute-prior, gray-world-dealiasing variant as an ablation; do not assume exact continuous equivariance from Fourier convolution alone.
4. **Same predictor, simple versus learned risk scores**: constant/source-order control, invariant candidate disagreement, and the held-out-residual head. For FFCC, include posterior concentration/covariance as a score if available. Compare risk at matched coverage and coverage at frozen source-calibrated thresholds.
5. **Deterministic algebra stress test**, with fixed masks, gains applied before the model, and ground truth transformed consistently. Report identity defects separately for ideal floating-point gains and a clipping/quantization stress track. These transformed samples are synthetic perturbations of real images; they do not create independent test subjects or cameras.

Select models and hyperparameters on source validation; lock them before target evaluation. Report per-camera recovery and reproduction summaries, selective reproduction risk, achieved coverage, uncertainty, CPU and RTX 4060 8 GB cost. A success requires real held-out-camera benefit at comparable cost; passing an algebra identity alone is insufficient. No angular result establishes facial CIELAB accuracy.

## Adoption ledger

| Asset | Code terms verified on owner page | Weights and data |
|---|---|---|
| Original [Google FFCC](https://github.com/google/ffcc/blob/master/LICENSE) | Apache-2.0; MATLAB runtime and the separate [minFunc dependency](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) need their own adoption review | No checkpoint/data clearance inferred from code license; retrain only on cleared source data |
| [Cotogni/Cusano offset-equivariant](https://github.com/claudio-unipv/offset-equivariant/blob/main/LICENSE) | MIT; owner repository describes PyTorch layers and CIFAR examples | No illuminant checkpoint or NUS rights cleared; no full 2024 implementation reproduction verified |
| IFFCC, uncertainty paper, CSNet, 2026 normalization wrapper | No code asset/license verified for adoption in this review | **COMMERCIAL USE NOT CLEARED** for any unverified code, weights or data |

For other named assets, retain the separate code/weight/data ledger in `public_color_prior_art.md`. Citations establish scientific overlap; they do not grant asset rights.
