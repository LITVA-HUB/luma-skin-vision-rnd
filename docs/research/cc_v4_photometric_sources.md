# CC V4 photometric supervision: primary-source review

Evidence cutoff: **2026-09-10**. **RESEARCH NOTE; NO MODEL OR DATA WAS DOWNLOADED, NO TRAINING WAS RUN, AND NO RESULT BELOW IS A LOCAL REPRODUCTION.** This note covers the physics/intervention thread in [`cc_v4_research_plan.md`](cc_v4_research_plan.md). It proposes experiments; it does not select the final V4 design or establish novelty.

## Decision

The most defensible new supervision available now is a **continuous action-conditioned reproduction-risk field trained on real, public illuminant ground truth**. For each real linear-RGB training image, evaluate many known candidate white-balance actions against its measured illuminant and train a compact model to predict the conditional risk of each action. Positive diagonal interventions create additional *exactly transformed labels* under stated assumptions; they do not create new independent scenes. A larger train-only teacher may smooth or enrich the field, but the measured illuminant remains the authority.

Do not make intrinsic decomposition, Retinex reconstruction, a diffusion restoration, a colorized image, or a cheap heuristic into target truth. Current intrinsic work reinforces why: single-image material/illumination separation remains underdetermined, and recent systems resolve it using strong learned priors, multiple illumination observations, privileged LiDAR, synthetic decompositions, or controlled hardware. Those mechanisms are useful as constraints and failure probes, not as replacements for illuminant ground truth.

The proposed field is **close prior art**, not a safe novelty claim. The 2020 multi-hypothesis method already corrects an image with candidate illuminants, scores each corrected image with a shared compact CNN, and combines the posterior. CLCC already synthesizes raw-domain illuminant changes and supervises illuminant-dependent representations. SAFE already searches a per-image continuous color-space action against ground-truth angular error and trains a student to predict that action. VLM-CC already performs iterative white-balance correction with residual-cast feedback. The narrower untested hypothesis is that directly learning a continuous, nonnegative, source-grounded reproduction-risk surface, including its lawful action derivatives and equivariance, yields better compact selection and abstention than these controls.

## Fresh and closest sources

Dates below are publisher or primary-repository dates, not search-engine dates. URLs were accessed on 2026-09-10.

| Primary source | What it establishes | Consequence here |
|---|---|---|
| Feng et al., [*Accelerated Self-Supervised Multi-Illumination Color Constancy With Hybrid Knowledge Distillation*](https://ieeexplore.ieee.org/document/11051054/), TPAMI 47(10), published **2025-06-25**, DOI `10.1109/TPAMI.2025.3583090` | Three phases: self-supervised pretraining, supervised color-constancy fine-tuning, and hybrid distillation. Light-normalization and grayscale-colorization pretexts train Transformer and U-Net teachers; their features are aligned to a CNN student. | Strong-teacher-to-compact-CNN transfer and color-specific pretexts are already direct color-constancy prior art. Distillation itself cannot be the V4 contribution. No official code, checkpoint license, or training-data ledger was located in the bounded search; imported artifacts are **COMMERCIAL USE NOT CLEARED**. A from-scratch teacher on already cleared data is the clean route. |
| Lee et al., [SAFE](https://arxiv.org/abs/2608.13967), arXiv v1 submitted **2026-08-14**; [project page](https://ntuneillee.github.io/research/safe/) | A compact feature model addresses pure-color ambiguity using scene-conditioned cue modulation. Its learned color space is bootstrapped by freezing a teacher, optimizing a per-image color axis for 80 steps against ground-truth angular error, then training a 12k-parameter predictor to imitate that oracle. | Very close to “optimize an action with a teacher, then predict it.” The proposed risk field must compare with a SAFE-style oracle-action/student control. SAFE searches a normalization axis rather than directly learning the full candidate-correction risk curve, but that distinction alone does not establish novelty. This is a recent preprint, not yet treated as peer-reviewed evidence here. |
| Hernández-Juárez et al., [*A Multi-Hypothesis Approach to Color Constancy*](https://openaccess.thecvf.com/content_CVPR_2020/html/Hernandez-Juarez_A_Multi-Hypothesis_Approach_to_Color_Constancy_CVPR_2020_paper.html), CVPR **2020-06**; [author code](https://github.com/huawei-noah/multi_hyp_cc) | Selects data-driven illuminant candidates, corrects a 64×64 image with every candidate, uses one shared 22.8k-weight CNN to estimate achromatic-light likelihood, combines likelihood and learned prior into a posterior, and regresses the final illuminant. | Closest correction-evaluator baseline. Continuous actions, direct reproduction-risk supervision, cached features, or gradients are incremental differences until a careful literature/patent review and matched experiments show otherwise. Its camera-specific candidate set and learned illuminant prior also supply negative controls. |
| Lo et al., [CLCC](https://arxiv.org/abs/2106.04989), CVPR 2021 / arXiv submitted **2021-06-09** | Explains that standard illumination-invariant contrastive learning conflicts with illuminant estimation. It builds supervised hard pairs and raw-domain novel-illuminant images. Its simplified WB augmentation first white-balances with one illuminant and applies inverse white balance for another; intensity/noise perturbations preserve the label. | Exact transformed-target training in linear raw RGB is established. V4 should use the intervention law to supervise a risk field rather than claim raw color augmentation or illuminant-dependent contrastive learning as new. It also supplies the strongest warning against semantic color-jitter invariance. |
| Cotogni and Cusano, [*Illuminant Equivariant Networks for Computational Color Constancy*](https://link.springer.com/chapter/10.1007/978-3-031-72845-7_18), CCIW held **2024-09-25–27**, publisher online **2024-10-18** | Direct task-level positive-diagonal illuminant equivariance. The earlier [general construction](https://arxiv.org/pdf/2207.00292), Neurocomputing 2022, supplies the normalization/predict/restore identity. | Equivariance under diagonal gain and anchor-normalized residual prediction are already known. V4 can use the law as supervision and a test oracle; it must not claim the law or wrapper as its contribution. |
| Li et al., [VLM-CC](https://openaccess.thecvf.com/content/CVPR2026/html/Li_White-Balance_First_Adjust_Later_Cross-Camera_Color_Constancy_via_Vision-Language_Evaluation_CVPR_2026_paper.html), CVPR **2026-06** | Iteratively applies a current white-balance estimate, renders pseudo-sRGB, asks a LoRA-tuned VLM for residual red/green/blue cast, and updates the illuminant. | Candidate correction followed by learned residual evaluation is current direct prior art. V4 may replace the VLM with a compact continuous risk field, but “white-balance then judge” and iterative feedback are not new. Its foundation-model inference is outside the deployment contract. |
| Liu and Wei, [*Color Constancy From a Pure Color View: An Edge-Aware Algorithm for a Wider Application*](https://onlinelibrary.wiley.com/doi/10.1002/col.70036), first published **2025-12-29**; Lee et al., SAFE above | PolyU Pure Color V2 contains 1,271 pure/near-monochromatic images. SAFE states the core ambiguity explicitly: a white illuminant on a red surface can match a red illuminant on a white surface. | A single-image field must represent irreducible ambiguity and may be confidently wrong if it learns only a source prior. Add a pure-color/low-chroma-diversity stratum and require risk to increase or abstention to trigger. The PolyU V2 repository has no located license file, so its images/weights are **COMMERCIAL USE NOT CLEARED** and are not adopted here. |
| Careaga and Aksoy, [*Colorful Diffuse Intrinsic Image Decomposition in the Wild*](https://yaksoy.github.io/papers/TOG24-ColorfulShading.pdf), TOG 43(6), **2024-12**, DOI `10.1145/3687984`; [project](https://yaksoy.github.io/ColorfulShading/) | Uses the richer linear model `I = A_d * S_d + R`: colorful diffuse shading plus a non-diffuse residual. It trains staged subproblems on eight synthetic datasets plus real multi-illumination data, and explicitly shows generative decompositions can warp faces/text, shift colors, or bake shading into albedo. | A grayscale Retinex model is too weak under colorful/mixed illumination, while the richer model is even less identifiable. Any auxiliary intrinsic head must remain a regularizer and be ablated. The owner code/models are academic-use-only and the method is patent-pending; they are **not eligible** as a production teacher without separate permission. |
| Djeghim et al., [SAIL](https://openaccess.thecvf.com/content/WACV2026/html/Djeghim_SAIL_Self-supervised_Learning_of_Lighting-Invariant_Representations_from_Real_Images_with_WACV_2026_paper.html), WACV **2026-03-06–10**, IEEE record added **2026-05-05** | Repurposes a latent diffusion prior for self-supervised relighting and latent intrinsic decomposition on unlabeled multi-illumination real images. The paper still calls in-the-wild single-view albedo estimation ill-posed and reports reflection/consistency limitations of earlier self-supervision. | Useful evidence that repeated real scenes can constrain illumination invariance, but the result is an albedo-like representation rather than measured illuminant truth. Diffusion-scale training, unavailable/unclear artifact terms, and the no-foundation-deployment requirement rule it out as the primary route. |
| Sato et al., [LIET](https://openaccess.thecvf.com/content/WACV2025/html/Sato_Unsupervised_Single-Image_Intrinsic_Image_Decomposition_with_LiDAR_Intensity_Enhanced_Training_WACV_2025_paper.html), WACV **2025-02** | Uses paired RGB and LiDAR intensity only during training. A separate LiDAR encoder and grayscale albedo-alignment loss transfer an illumination-insensitive infrared reflectance cue; inference uses RGB only. | This is a legitimate *privileged training sensor* pattern, but LiDAR intensity is wavelength-, distance-, angle-, and material-dependent and supplies no RGB hue. It is not an illuminant label. The [NTT-IID license](https://github.com/ntthilab-cv/NTT-intrinsic-dataset/blob/main/LICENSE) permits internal noncommercial evaluation only and restricts modification/transfer, so it is **not eligible** for this commercial R&D training path. |
| Choi et al., [*A Real-world Display Inverse Rendering Dataset*](https://openaccess.thecvf.com/content/ICCV2025/html/Choi_A_Real-world_Display_Inverse_Rendering_Dataset_ICCV_2025_paper.html), ICCV **2025-10**; [project](https://michaelcsj.github.io/DIR/) | A calibrated LCD and stereo polarization cameras capture 16 real objects under 144 OLAT display patterns, with RAW polarization, light calibration, and scanned geometry. Linear combinations synthesize controlled display illumination. | Strong evidence that real physical transform supervision is possible when illumination and capture are instrumented. It is specialized multi-view/object photometric stereo, not an in-the-wild global-illuminant benchmark. The data card says CC BY 4.0, but the author GitHub has no license file and the card/page release state was inconsistent at review time. Do not adopt it without a release-level terms/hash audit. |
| Kim et al., [ParamISP](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_ParamISP_Learned_Forward_and_Inverse_ISPs_using_Camera_Parameters_CVPR_2024_paper.html), CVPR **2024-06**; Cheng et al., [post-ISP WB editing](https://openaccess.thecvf.com/content/WACV2026/html/Cheng_Perception-Inspired_Color_Space_Design_for_Photo_White_Balance_Editing_WACV_2026_paper.html), WACV **2026-03** | ParamISP learns forward/inverse RAW↔sRGB mappings conditioned on ISO/exposure, because camera pipelines vary. Cheng et al. separately describe sensor RAW as linear in radiance and sRGB correction as a post-ISP editing problem with nonlinear, channel-entangling transforms. | The exact V4 diagonal law belongs to black-level-subtracted linear sensor RGB. Unknown-ISP/JPEG input is a separate task. An inverse ISP estimate cannot silently turn JPEG into known RAW or restore clipped information. |
| Hu et al., [ShaRP](https://arxiv.org/abs/2410.02057), submitted **2024-10-02**, ICML 2025; [project](https://wustl-cig.github.io/sharpwww/) | Uses an ensemble of degradation-conditioned MMSE restoration operators as a stochastic prior and derives a regularizer from their score functions. Different structured restoration tasks can provide a stronger prior than one Gaussian denoiser. | Inspires condition-dependent probes rather than an image-restoration target. A bank of lawful photometric actions can reveal estimator instability, but its dispersion is only a feature until calibrated against held-out real illuminant residuals. The released SISR repo/checkpoint has no located license file; **COMMERCIAL USE NOT CLEARED**. |
| Pereyra and Tachella, [*Equivariant bootstrapping for uncertainty quantification in imaging inverse problems*](https://proceedings.mlr.press/v238/pereyra24a.html), AISTATS **2024-05-02–04**; [MIT code](https://github.com/tachella/equivariant_bootstrap) | Uses known symmetries and replicated forward experiments to construct uncertainty regions for inverse imaging, with theory for linear estimators. | Supports using lawful intervention replicas as uncertainty evidence. It does **not** make augmentation dispersion a calibrated color-constancy risk or transfer its coverage theorem to nonlinear ISP, learned risk, or domain shift. Source-held-out calibration remains required. |
| Kouros et al., [*Unveiling the Ambiguity in Neural Inverse Rendering*](https://openaccess.thecvf.com/content/CVPR2024W/NRI/html/Kouros_Unveiling_the_Ambiguity_in_Neural_Inverse_Rendering_A_Parameter_Compensation_CVPRW_2024_paper.html), CVPRW **2024-06** | Perturbing one recovered scene property can be compensated by another while retaining plausible renderings; added geometry/material/illumination guidance is needed. | Reconstruction or correction consistency alone cannot identify physical reflectance and illumination. A V4 auxiliary decomposition that reconstructs the input can pass its loss with wrong factors. |
| Wu et al., [Measured Albedo in the Wild](https://measuredalbedo.github.io/), ICCP/arXiv submitted **2023-06-27** | Provides measured average RGB albedo for homogeneous regions in 888 images/46 indoor scenes and evaluates albedo intensity, chromaticity and texture. The method uses a gray card and linearized RAW capture; it is not a CIE Lab instrument benchmark for every surface. | A useful future physical surface-color lead, but separate from global illuminant estimation. The project licenses the *website source* CC BY-SA, while no dataset license was found on the owner page; data use is **COMMERCIAL USE NOT CLEARED**. Do not manufacture Delta E from it. |
| Land and McCann, [*Lightness and Retinex Theory*](https://opg.optica.org/josa/abstract.cfm?uri=josa-61-1-1), published **1971-01-01**, DOI `10.1364/JOSA.61.000001` | Foundational spatial lightness/reflectance account; it seeks a correlate of reflectance from spatial comparisons rather than direct flux. | Retinex is historical motivation for relational cues, not evidence that a learned single-image decomposition recovers physical albedo or illuminant uniquely. |

## Exact transformation contract

Let `x[p] in R^3_+` be black-level-subtracted, unsaturated **linear sensor RGB** for valid pixel `p`, and let `g in R^3_+` be the measured global illuminant direction in the same camera RGB space. Define centered log chroma

```text
clr(v) = log(v) - mean(log(v)) * [1,1,1]
gamma = clr(g)
```

and represent a candidate illuminant by a two-degree-of-freedom action `a` with zero channel sum, `e(a) proportional exp(a)`. Candidate correction is

```text
C_a(x)[p] = x[p] / e(a)                 # componentwise; common scale is irrelevant
```

For a known positive diagonal intervention `D = diag(exp(d))`, take `sum(d)=0` without loss of generality:

```text
x' = D x
g' = D g
gamma' = gamma + d
a' = a + d
C_(a+d)(D x) proportional C_a(x)
```

The established reproduction error is

```text
rho(g,e) = angle(g/e, [1,1,1])
```

so the exact joint-action identity is

```text
rho(Dg, De) = rho(g,e).
```

This provides real supervised points `(D x, a+d, rho(g,e(a)))` without inventing a surface target. It also provides an exact consistency oracle for predictions. The identity is algebraic for positive channels; interpreting `D x` as a *new physical illuminant capture* additionally assumes a single global illuminant, a diagonal/von-Kries sensor-space change, fixed exposure/geometry/reflectance, and no clipping or black-level error.

Other admissible transformations have narrower contracts:

| Transformation on linear RGB | GT/action law | Valid only when |
|---|---|---|
| Common exposure `x' = alpha x`, `alpha > 0` | Illuminant direction and action unchanged | Valid-pixel support is fixed; neither shadows nor highlights cross censoring thresholds; black level is already removed. |
| Flip or pixel permutation | `g` unchanged | Global illumination is uniform and the calibration-object mask remains valid. |
| Crop | `g` unchanged only as a dataset label | The original label is truly global. Cropping a mixed-light scene can change the dominant local illuminant, so do not call it an exact physical intervention. |
| Same linear spatial operator in every channel, such as blur before subsampling | `g` unchanged and it commutes with `D` | Kernels/boundaries are channel-identical, positive support is retained, and the operation is after demosaic or defined consistently on the mosaic. |
| Additive sensor noise | `g` unchanged in the generative model, not pointwise | Noise parameters are known and independent of the label. It does not produce an exact corrected-image equality. |
| Full 3x3 color mixing `A x` | Algebraically `g' = A g`, but no reproduction-risk invariance | Output remains positive and `A` models an actual sensor/color-space mapping. This cannot stand in for a new camera across all reflectances, and V3 already found that a stronger GL construction can hurt. |
| Hue/saturation/contrast, local tone mapping, JPEG recompression | **No exact illuminant law** | Use only as a separately labeled post-ISP robustness stressor. Never use the old `g` as if the action preserved it. |

For unknown-ISP input `y = F(x)`, generally `F(Dx) != D F(x)` because `F` can include its own white balance, a full CCM, channel curves, local tone mapping, gamut mapping, clipping, sharpening, and compression. A diagonal change applied to JPEG pixels is a post-capture edit, not a known change of the sensor illuminant. V4 should therefore have two declared domains:

1. **Primary scientific domain:** linear sensor RGB with the exact laws above and angular illuminant evaluation.
2. **Exploratory post-ISP domain:** unknown/JPEG processing, evaluated on paired or otherwise measured targets only. Inverse-ISP output is an estimate with lost information, not recovered truth.

## Proposed mechanism: action-conditioned conditional risk

For a labeled real training row `(x,g)`, sample candidate actions `a` around several fixed cheap anchors and across a prespecified broad log-chroma box. Define a bounded smooth reproduction loss

```text
s(g,a) = 2 * (1 - cos(rho(g,e(a)))) >= 0.
```

It has the same per-row ordering as reproduction angle, avoids the angular derivative singularity at zero, and obeys `s(Dg,a+d)=s(g,a)`. Train a nonnegative field

```text
q_theta(x,a) ~= E[s(g,a) | x,a]
```

with `q_theta = softplus(raw_theta)`. The expectation is essential: pure-color and other metameric/scene ambiguities can make several illuminants plausible for the same observed image. For a labeled point the target is computed exactly from `(g,a)`; no pseudo-ground-truth image, Delta E, albedo, or semantic judgment is used.

A compact implementation candidate is one image encoder cached once, plus an action decoder:

- image branch: a small 1–5M-parameter CNN over linear RGB/log-chroma plus fixed valid-mask channels;
- fixed cheap cues: Gray World, Shades of Gray, Gray Edge, saturation/black fractions, chroma entropy and dispersion;
- action branch: sinusoidal or polynomial features of the 2D centered-log-gain action and analytic corrected-cue transforms;
- fusion: FiLM or a small cross-MLP that outputs `q_theta(x,a)` and optional epistemic scale;
- inference: evaluate a fixed coarse action lattice around the cheap anchors using the cached image state, refine the best few actions with two bounded gradient steps, and return the minimum-risk illuminant plus source-calibrated abstention score.

This differs operationally from rerunning a full CNN for every corrected thumbnail, but the comparison must include the original multi-hypothesis pattern. A second matched control should feed actual candidate-corrected 64×64 thumbnails through the same tiny scorer. Any speed/accuracy claim needs local measurement.

The proposed training objective is

```text
L = L_point + lambda_grad L_grad + lambda_eq L_eq
    + lambda_rank L_rank + lambda_KD L_KD.

L_point = Huber(q_theta(x,a), s(g,a))
L_grad  = Huber(grad_a q_theta(x,a), grad_a s(g,a))
L_eq    = |q_theta(Dx,a+d) - q_theta(x,a)|
L_rank  = max(0, margin - sign(s2-s1) * (q2-q1))
```

`L_grad` uses the analytic/autodiff derivative of the smooth chordal target, not a finite-difference image score. Omit/clip it only at declared numerical boundaries. `L_rank` samples two actions on the *same real row* and uses their exact target ordering. `L_KD` is optional teacher-to-student field matching and must never outweigh `L_point`; a teacher disagreement cannot overwrite measured GT.

For selection, minimize `q_theta(x,a)` inside the prespecified action box. Report actual reproduction and recovery angular errors of the selected illuminant. Do not report `q_theta` itself as degrees, expected Delta E, a confidence guarantee, or calibrated failure probability. Fit any degree-error/risk calibration only from scene/subject-held-out source residuals, then freeze it before outer-camera evaluation.

### Legitimate strong-teacher route

The clean teacher is trained **from scratch** on the same legally reviewed real illuminant-GT training rows, with every evaluation camera absent from fitting, selection, risk calibration, and teacher-label generation. It may be a larger corrected-thumbnail CNN/Transformer or an ensemble of direct, FFCC-inspired, and action-field estimators. Dense teacher fields can teach smoothness between sampled actions, while exact `s(g,a)` anchors every sampled action.

Use out-of-fold teacher predictions for any residual-risk or calibration target. In-fold logits/features are acceptable only as ordinary distillation regularization against labels the student already sees; they are not new evidence. A teacher trained on the outer camera, target batch, or its labels leaks the evaluation domain even if the student never sees those rows directly.

Cheap experts are also legitimate as **anchors and proposals**: GW/SoG/Gray Edge can define candidate centers, and their exact error against training GT can be an input target. They are not pseudo-GT. TPAMI 2025 shows that Transformer/U-Net-to-CNN hybrid distillation is already a direct technique; the experiment must isolate whether the continuous physical risk target adds value beyond ordinary logit/feature distillation.

Avoid third-party pretrained teachers in the first decisive experiment. Their checkpoint terms, training-image rights, and distilled-output implications are separate from code licenses. A from-scratch teacher is affordable at thumbnail resolution on the existing RTX 4060 path and eliminates that unresolved dependency.

## Why intrinsic and restoration constraints stay auxiliary

An optional auxiliary branch may predict low-frequency illumination/shading structure and enforce that corresponding features agree under `(x,a)` and `(Dx,a+d)`. It must not be supervised from a generic intrinsic model. Reconstruction `x ~= A*S+R`, Retinex gradients, total variation, or corrected-image similarity can regularize representation, but each admits compensating wrong factors. Colorful/mixed lighting, specularity, fluorescence, interreflection, non-Lambertian materials, spatially varying light, sensor metamerism, and unknown ISP all violate the simplest factorization.

Restoration-conditioned probes can still inform *risk*: apply label-preserving exposure/noise/spatial operators with known contracts, invert each prediction back to the original action coordinates, and measure field disagreement. ShaRP motivates conditioning on a distribution of degradations; equivariant bootstrapping motivates replicated symmetry experiments. Neither turns dispersion into a guarantee. Calibrate its relationship to error on held-out source rows and retain a no-probe risk baseline.

## Fatal confound and falsification tests

These are gates, not optional visualizations.

1. **Correction-evaluator prior art.** Compare against the 2020 multi-hypothesis structure with the same candidate count, image size, backbone budget, augmentation and training rows. Compare a likelihood/posterior head, direct point estimator, discrete supervised action-cost head, and continuous action field. If only more candidates or more compute helps, reject the mechanism claim.
2. **SAFE-style oracle/student.** Optimize one per-image action against a frozen teacher and train a student to predict it. If this matches the field, the dense field/gradient mechanism has no demonstrated benefit.
3. **Ordinary distillation.** Distill the same strong teacher's point illuminant or logits into the same student. The risk-field experiment must improve over it at matched data, teacher, parameters and optimization budget.
4. **Action-only shortcut.** Train/evaluate a model with the image zeroed or shuffled within camera. Use identical action sampling for every row. If the field retains useful accuracy, it is learning source illuminant priors rather than image evidence.
5. **Transform-parameter shortcut.** Hide `d` from one branch and test unseen intervention magnitudes/directions. Joint-shift equivariance must hold after undoing the known transform, not merely because the network reads `d`.
6. **Absolute-anchor failure.** A model can satisfy `q(Dx,a+d)=q(x,a)` while its entire field is shifted. Require exact GT point loss, report the selected absolute angular error, and test a constant/shifted field negative control.
7. **Synthetic multiplicity.** Weight or sample by original real row so 100 actions do not count as 100 independent scenes. Confidence intervals and splits use physical scene/group units. Report action expansion as augmentation only.
8. **RAW-versus-JPEG boundary.** Run exact-law numerical tests on unclipped linear arrays. Separately apply the same nominal gains after gamma/tone mapping/JPEG and measure identity failure. Any post-ISP benefit is empirical editing robustness, not proof of the RAW law.
9. **Censoring/support changes.** Track pixels crossing black/saturation thresholds under every `D` or exposure action. Compare fixed-mask lawful training with recomputed-mask training. Reject results driven by clipping halos or mask/card leakage.
10. **Mixed illumination/crops.** Stratify known or suspected multi-illuminant scenes. Crop consistency is not enforced there. A global label and low field risk must not be presented as correct local white balance.
11. **Pure-color ambiguity.** Evaluate low chroma-entropy and near-monochromatic strata, with the camera held out. Require worse predicted risk/abstention when absolute error rises. If the model remains confident because of learned scene semantics, reject the selective-risk claim.
12. **Camera/metadata leakage.** Remove filenames, camera IDs, paths, EXIF and per-camera normalization from the strict model. Compare against an explicit camera-aware upper bound. Action grids and priors are fit from source training only.
13. **Duplicate-scene leakage.** Keep reference-hash and physical-scene groups within one role. Teacher, student, risk calibrator and evaluation splits inherit the V3 grouping exclusions.
14. **Cheap-cue mimicry.** Regress the selected output and risk using only GW/SoG/Gray Edge features. If the field does not beat this control, retain the cheaper method.
15. **Risk learns intervention magnitude.** At fixed true error, test whether risk rises simply with distance from the anchor, clipping fraction or noise level. Calibrate on held-out clean source rows and report clean and stressor results separately.
16. **Teacher inheritance.** Compare GT-only student, teacher-distilled student and teacher-only performance on held-out source groups. A student cannot be described as more physical because it imitates a strong but biased teacher.
17. **Intrinsic compensation.** For any auxiliary `A*S+R` head, perturb one component and optimize another as in the inverse-rendering ambiguity analysis. If reconstruction stays stable while factors move, do not interpret components physically.
18. **No fabricated surface metric.** The only current target is illuminant recovery/reproduction geometry. No CIELAB/Delta E result is emitted unless a separate dataset supplies aligned measured surface values, observer/illuminant/geometry, and a fixed sensor-to-XYZ pipeline.

## Rights and adoption ledger

No external artifact is adopted by this note. Status is conservative and separates paper access, code, weights and data.

| Candidate asset | Code terms | Weight terms | Data terms / decision |
|---|---|---|---|
| Proposed V4 implementation and from-scratch teacher | New project-owned code | New weights, but downstream distribution must retain training-data lineage | SimpleCube++ owner repository states **CC BY 4.0**. INTEL-TAU original registry states **CC BY-SA 4.0**; models carry explicit source lineage and no claim of unrestricted proprietary redistribution clearance. Preserve attribution/ShareAlike analysis. These are the only proposed real-GT sources at this stage. |
| TPAMI 2025 hybrid-distillation artifacts | No official author code located | No official checkpoint/license located | Paper benchmark names do not grant data rights. Concept may be reimplemented; artifacts remain **COMMERCIAL USE NOT CLEARED**. |
| Multi-hypothesis CVPR 2020 | Author repo [BSD-0](https://github.com/huawei-noah/multi_hyp_cc/blob/master/LICENSE) | Paper uses an ImageNet-pretrained first layer; checkpoint-specific provenance was not cleared here | Dataset rights remain separate. Use as a reference/reimplementation or complete a checkpoint ledger before import. |
| SAFE / PolyU Pure Color V2 | No SAFE code release located; arXiv's publication license and the project website's CC BY-SA footer do not license an implementation | None located | ePCC repository links the dataset but contains no located license file. **COMMERCIAL USE NOT CLEARED**; do not acquire or train on it under this plan. |
| CLCC | Paper HTML is CC BY 4.0; the author code URL returned 404 at review | Not verified | NUS/Gehler rights are separate. Reimplement only the documented law if needed; do not assume paper license covers code/data. |
| Colorful Intrinsic | Owner repository says academic use only; method is patent-pending | Models covered by the same restrictive notice | Training datasets have separate terms. Exclude code/weights as a commercial teacher unless separately licensed. |
| SAIL / latent diffusion | Project-page source found, but no released method-code license located | Base latent-diffusion and fine-tuned weight terms not audited | BigTime and other image sources require their own rights audit. Exclude from the first path. |
| NTT-IID / LIET | No LIET author code located | None located | NTT dataset license is noncommercial evaluation-only and restricts transfer/modification. Exclude. |
| DIR | Author GitHub has no license file | No weights needed for the optimization baseline | Hugging Face card labels data CC BY 4.0, but release/card state was inconsistent and the dataset is not color-constancy GT. Re-audit exact files and terms before any use. |
| ShaRP | Released SISR/MRI repos have no located license file | Downloadable SISR checkpoint has no separate located terms | Datasets/base restoration dependencies are separate. Use the principle only; do not import artifacts. |
| Equivariant bootstrap | Author code is MIT | Downloaded comparison checkpoints mentioned by the repo have separate terms | No candidate data adoption. Theory does not confer calibration on V4. |
| MAW | Evaluation code license not verified in this review | Not applicable | Project page licenses only its website source and does not state dataset terms. **COMMERCIAL USE NOT CLEARED**. |

## Decisive experiment recommendation

Keep this as one candidate for root-level comparison rather than freezing it prematurely. The smallest informative source-only experiment is:

1. Lock a scene/group-disjoint source train/validation/calibration split before action generation.
2. Use the existing real illuminant GT only. Generate positive diagonal actions within a predeclared no-clipping range and count physical rows, not actions.
3. Train matched `<5M` models from scratch: direct estimator, discrete multi-hypothesis correction scorer, continuous field without gradients/equivariance, and the full point+gradient+equivariance field.
4. Add cheap-anchor-only and action-only negative controls. Run one seed for mechanism screening; repeat only a surviving comparison.
5. Select architecture/checkpoint and fit any abstention calibration on source roles only. Then freeze before a genuinely outer-camera protocol.
6. Report full reproduction/recovery distributions, risk/coverage, pure-color proxy strata, boundary/clipping failures, parameter count, latency and VRAM. Label all diagonal-action rows **controlled augmentation**, never independent benchmark evidence.

Promotion requires the field to improve source validation and outer-camera selective reproduction risk over the matched direct and multi-hypothesis controls without relying on camera metadata, a restricted teacher, or post-ISP label assumptions. Failure is scientifically useful: it would show that exact counterfactual supervision adds density but not identifiable scene evidence.
