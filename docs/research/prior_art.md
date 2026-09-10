# Public color-constancy revision — 2026-09-10

## V3 extension: color frames, posterior geometry and FFCC

The newest [source-grounded review](cc_v3_prior_art.md) identifies explicit GL-frame/canonicalization ancestry, the difference between canonical and camera-space likelihoods, conditioning/continuity limitations, and the pinned Apache-2.0 FFCC formulas. Full-frame graph canonicalization is not a new theorem. The [real source screen](../benchmarks/cc_v3_report.md) is negative: full-frame Proposed loses to its matched direct graph and does not beat Shades of Gray. [The next hypothesis](cc_v3_revision_decision.md) retains global color information and requires new measurement. These results do not replace the limited positive V2 camera-transfer evidence below.

The public review is [public_color_prior_art.md](public_color_prior_art.md), including FC4, Reweight-CC, C5, CCMNet, uncertainty2025, VLM-CC2026, GC3, GCC and BRE. The current bounded extension is [cc_v2_prior_art.md](cc_v2_prior_art.md). Both narrow rather than establish novelty. The prior facial-specific analysis below is retained as historical context.

## V2 extension: known equivariance, measured selective transfer

**Status: V2 measured.** The source-selected SoG residual+combined head reaches fresh risk80=3.805° versus matched direct C+=5.558° on384 held-out-camera images, while source and Canon results regress and cheap GW+ridge remains competitive. This supports a bounded empirical component effect, not wrapper novelty; see [measured V2 report](../benchmarks/cc_v2_report.md). Investigate source-trained compact CNN residual correction after an unnormalized channelwise Gray World/Shades of Gray anchor, followed by a source-fitted selective reproduction-risk head using normalized scene context and relative log-ratios. The normalization/prediction/restoration wrapper is already covered by Cotogni and Cusano's [2022 construction, Eq. (19)](https://arxiv.org/pdf/2207.00292), DOI `10.1016/j.neucom.2022.06.118`. Their [CCIW 2024 illuminant-equivariant work](https://link.springer.com/chapter/10.1007/978-3-031-72845-7_18), DOI `10.1007/978-3-031-72845-7_18`, is further direct task-level overlap; its abstract/metadata, not full chapter equations, were accessible. No wrapper novelty is claimed.

The [v2 note](cc_v2_prior_art.md) derives the precise positive-diagonal-gain identity, distinguishes componentwise equivariance from scalar homogeneity, and records failures caused by floors, changing masks, clipping and non-diagonal camera responses. Invariant risk features align with reproduction error; recovery angle is not gain-invariant. Neither identity proves camera transfer or calibration. [Original FFCC](https://github.com/google/ffcc) is an Apache-2.0 compact learned comparator; its complete model includes absolute illuminant priors, so a no-prior variant must be identified separately. IFFCC2025, uncertainty2025, CCMNet2025, BRE2026, VLM-CC2026, CSNet2026 and a 2026 normalization-wrapper preprint further narrow the claims; citations, access limits and separate code/weight/data terms are in the v2 note.

The user's A/B/C candidates remain intact: A concerns downstream-color-risk selection (currently only an angular proxy); B concerns cheap hypotheses, disagreement and contextual residual correction; C concerns compact single-image unseen-camera estimation. V2 tests a known B mechanism within C and evaluates the A proxy against strong matched C+ controls. The [v1 negative findings](negative_results.md) and [measured benchmark](../benchmarks/public_benchmark_report.md) remain evidence: the mixture's selective advantage was not established, and Sony transfer lost to Gray World. The historical synthetic loss to A2 remains separate. V2 does not erase, relabel or retrospectively pass those results. Any benefit must be measured with source-only selection and risk calibration, frozen target evaluation, matched compute and preserved negative variants; facial CIELAB validity remains deferred.

---

Historical synthetic/facial-stage material follows.

# Prior art — evidence review, 2026-09-10

This is a targeted primary-source literature review, not an exhaustive patent search or independent replication. Publication results below belong to their authors. NOT EXTRACTED means the value was not verified in this review; it does not mean the paper lacks it. NOT MEASURED describes Luma results. No paper, dataset, or checkpoint listed here has been reproduced on Luma data. Sources were checked on 2026-09-10; preprints can change.

## Close analogues

### 1. Deep White-Balance Editing — Afifi and Brown, CVPR, June 2020

Problem: correcting ISP-rendered sRGB white balance. Method/model: multitask encoder-decoder predicting white-balanced outputs, including indoor/outdoor settings; nonlinear ISP motivates learned correction. Data/GT: Rendered WB paired images and rendered reference WB, not instrumented cheek Lab. Metrics: image correction error including color error; numerical results and compute hardware NOT EXTRACTED. Authors report improvement over their earlier KNN approach. [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Afifi_Deep_White-Balance_Editing_CVPR_2020_paper.html), [code/data instructions and terms](https://github.com/mahmoudnafifi/Deep_White_Balance).

Code: expressly research-only, noncommercial. Weights: no independent commercial grant verified. Data: separate Rendered WB terms need review. Commercial use NOT CLEARED. Overlap: multiple plausible WB outputs and nonlinear JPEG correction substantially predate Luma. Difference: no verified instrument-specific selective cheek measurement. Use as literature comparator; do not import weights or generate commercial training labels from it without clearance.

### 2. C5 — Afifi et al., preprint 2020-11-24; ICCV 2021

Problem: unseen-camera raw illuminant estimation. Model: hypernetwork generates convolutional color-constancy parameters from image color statistics and additional unlabeled images from the target camera. Data/GT: multiple camera datasets including NUS and Gehler-Shi; reference illuminant, not facial reflectance. Metric: angular illuminant error; accuracy numbers NOT EXTRACTED. Authors report approximately 7 ms GPU / 90 ms CPU and 2 MB; exact hardware NOT EXTRACTED, so these are not deployment forecasts. [Paper](https://arxiv.org/abs/2011.11890), [reference repository](https://github.com/mahmoudnafifi/C5).

Code: Apache-2.0 in [LICENSE](https://github.com/mahmoudnafifi/C5/blob/main/LICENSE). Weights: checkpoint-specific grant not separately verified. Data: each source dataset requires separate review. Full commercial pipeline NOT CLEARED. Overlap: camera adaptation and test-time color ambiguity. Its multi-image raw input is a different information regime from single JPEG; compare separately if used.

### 3. TRUST — Feng et al., preprint 2022-05; ECCV 2022

Disambiguation: here TRUST means *Towards Racially Unbiased Skin Tone Estimation via Scene Disambiguation*, not an unrelated confidence/trust-score method. Problem/model: facial albedo estimation conditioned on face and whole-scene illumination, with balanced albedo representation. Data/GT: FAIR scan-based albedo benchmark, evaluated using ITA and skin-tone balance; not this project's contact-instrument cheek protocol. Numerical results and hardware NOT EXTRACTED; authors report improved albedo accuracy/fairness. [Project](https://trust.is.tue.mpg.de/), [paper](https://arxiv.org/abs/2205.03962).

Code, weights and associated data: [license](https://github.com/HavenFeng/TRUST/blob/main/LICENSE) restricts use to noncommercial scientific research and explicitly excludes training algorithms for commercial use. COMMERCIAL USE NOT CLEARED, including teacher labels/distillation. Strong overlap: illumination/albedo ambiguity is already central to skin estimation. Luma cannot claim discovering it. Full-scene cues are a useful planned ablation, with own data and implementation.

### 4. Improving Fairness in Facial Albedo Estimation via Visual-Textual Cues — Ren et al., CVPR 2023

Problem/model: skin-tone-biased albedo; visual/textual guidance and albedo generation. Data/GT: SFHQ synthetic training; FAIR benchmark with 206 head scans; FFHQ qualitative examples. Metrics: ITA error and between-group bias; authors report best average ITA score and near-TRUST overall score. Numerical table values NOT EXTRACTED. Training hardware NOT EXTRACTED. [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ren_Improving_Fairness_in_Facial_Albedo_Estimation_via_Visual-Textual_Cues_CVPR_2023_paper.pdf).

Code license: NOT VERIFIED. Weight license: NOT VERIFIED. Dataset rights: SFHQ/FAIR/FFHQ separate, NOT CLEARED for this project. Overlap: synthetic lighting/albedo supervision and skin-tone robustness. Limitation for Luma: ITA/albedo reconstruction is not instrumented regional ΔE00, and synthetic appearance fidelity does not establish measurement trueness.

### 5. HUST: High-Fidelity Unbiased Skin Tone Estimation via Texture Quantization — Ran et al., ICCV 2025

Problem/model: faithful and less biased facial albedo; texture quantization. Data/GT: scan/albedo skin-tone evaluation lineage; exact dataset inventory, metrics table values and hardware NOT EXTRACTED. The paper explicitly builds on TRUST's balanced albedo and lighting context. [Primary paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Ran_HUST_High-Fidelity_Unbiased_Skin_Tone_Estimation_via_Texture_Quantization_ICCV_2025_paper.pdf).

Code: NOT VERIFIED. Weights: NOT VERIFIED. Data: NOT VERIFIED. COMMERCIAL USE NOT CLEARED. High-priority follow-up for full table extraction before a novelty assertion. Overlap weakens broad claims about unbiased intrinsic facial color; identity-preserving texture is outside Luma's non-identification target. No verified result for calibrated instrument-error rejection was extracted.

### 6. Smartphone tristimulus colorimetry for skin-tone analysis at common pulse oximetry anatomical sites — Burrow et al., preprint 2024-11-21; journal 2025-05-19

Problem/method: accessible instrument-compared skin color; controlled smartphone acquisition and RGB→Lab→ITA mapping. Data: four volunteers in preprint, finger/wrist anatomical sites. GT: professional tristimulus colorimeter (DSM-4 in journal). Metrics: ITA agreement; exact numerical errors NOT EXTRACTED. Authors report good correlation under low ambient light with flash disabled. Hardware: smartphone, DSM-4; precise phone/compute configuration NOT EXTRACTED. [Preprint](https://arxiv.org/abs/2411.13832), [journal record](https://doi.org/10.1117/1.BIOS.2.3.032504).

Code: no verified release. Weights: no learned model identified. Data: no commercial reuse grant verified. COMMERCIAL USE NOT CLEARED for artifacts. Strong overlap: instrument-based smartphone skin color is established. Limitation: small controlled sample, different anatomical sites, ITA rather than locked selective cheek ΔE00. It supports including a disciplined classical capture baseline before ML.

### 7. Region-Specific Calibration Achieves Excellent Inter-Device Reliability for Smartphone Dermatology — Kang and Kim, 2025-12-26; v4 2026-07-02

Problem/method: cross-device facial color consistency; global versus region-specific linear CCM in Lab. Data: 965 Korean subjects, DSLR/tablet/smartphone matched facial images. Reference: DSLR, explicitly not contact spectrophotometry. Metrics: ΔE00, ICC, anatomical/device variance. Published abstract: global CCM reduces color differences 61–74%; regional CCM yields MI ICC 0.95 and ITA ICC 0.93. Compute hardware NOT EXTRACTED. [Versioned paper](https://arxiv.org/html/2512.21988v4).

Code: release/license NOT VERIFIED. Weights: no neural checkpoint required; fitted CCM terms NOT VERIFIED. Data: commercial rights NOT VERIFIED; paper CC-BY does not license participant images. COMMERCIAL USE NOT CLEARED. This directly weakens claims that regional cross-camera calibration is new. Add a regional CCM baseline when paired data supports it; do not equate DSLR agreement with instrument trueness or generalization to unseen pipelines/populations.

### 8. Bridging Accessibility and Precision: Evaluating the Reliability and Validity of a Smartphone-Based Skin Colorimeter — Ouyang and Yang, 2026

Problem/method: facial smartphone colorimeter YLGTD; reported neural iterative correction with device/exposure parameters. GT: VISIA imaging system and DermaLab Combo spectrophotometer. Exact cohort, architecture, publication day, numerical agreement metrics and compute hardware NOT EXTRACTED: publisher access returned 403 and PMC challenge limited full extraction. [Publisher](https://doi.org/10.2147/CCID.S589014), [primary full-text record](https://pmc.ncbi.nlm.nih.gov/articles/PMC13050169/).

Code: NOT VERIFIED. Weights: NOT VERIFIED. Data: NOT VERIFIED. COMMERCIAL USE NOT CLEARED. This is very close problem-level prior art, including instrument comparison and device-conditioned neural correction. It must be fully reviewed before claiming distinctiveness. No absence of selective prediction is asserted from incomplete access. Luma must outperform a matched correction-plus-error baseline, not just naive pixels.

### 9. VLM-CC: White-Balance First, Adjust Later — Li et al., 2026-05-19; CVPR 2026

Problem/model: cross-camera raw color constancy. Iterative WB produces pseudo-sRGB; LoRA-tuned vision-language model evaluates residual cast, providing RGB-direction correction. Data/GT: cross-camera illuminant benchmarks; exact dataset list and quantitative metrics NOT EXTRACTED. Authors report improved cross-camera robustness; hardware NOT EXTRACTED. [Paper](https://arxiv.org/abs/2605.19613).

Code: release announced, license NOT VERIFIED. Weights: VLM and LoRA terms NOT VERIFIED. Data: NOT VERIFIED. COMMERCIAL USE NOT CLEARED. Overlap: explicit correction hypotheses and assessment of remaining ambiguity. It is not a reason to put an LLM into measurement; neither compactness nor instrument-target validity follows from perceptual color-cast feedback. It motivates testing hypothesis disagreement against standard correction quality signals.

### 10. SelectiveNet — Geifman and El-Yaniv, ICML 2019-06

Problem/model: prediction with rejection; shared representation, prediction/selection/auxiliary heads and coverage-constrained loss. Data/GT: classification and regression benchmarks; task labels, not colorimeter measurements. Metrics: selective risk at prescribed coverage; numerical results and compute hardware NOT EXTRACTED. Authors report better risk–coverage tradeoffs. [Paper and code link](https://proceedings.mlr.press/v97/geifman19a.html).

Code: linked but license NOT VERIFIED. Weights: NOT VERIFIED. Data: benchmark-specific, NOT VERIFIED. COMMERCIAL USE NOT CLEARED for imported artifacts. Overlap is fundamental: joint regression/rejection is established. A dedicated error head is not itself novel. A fair Luma comparison needs a conventional residual selector and ideally selective-loss baseline under the same backbone, data and tuning budget.

### 11. On Calibration of Modern Neural Networks — Guo et al., ICML 2017

Problem/method: classification probability miscalibration; post-hoc temperature scaling. Data/GT: image/document classification, including CIFAR/ImageNet-class experiments; correctness labels. Metrics: reliability diagrams, ECE and probabilistic scoring; numerical result table and hardware NOT EXTRACTED. Authors find temperature scaling effective on many tested classifiers. [Primary paper](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).

Code: license NOT VERIFIED. Weights: NOT VERIFIED. Data: source-specific NOT VERIFIED. COMMERCIAL USE NOT CLEARED for artifacts. Overlap: calibrated confidence is established. Limitation: temperature-scaled class probabilities cannot be relabeled expected ΔE00. A separately supervised event ΔE00≤tolerance can be calibrated, using subject-held-out residuals and a distinct calibration split.

### 12. Learn then Test — Angelopoulos et al., preprint 2021-10-03

Problem/method: distribution-free finite-sample risk control; learn candidate rules then test risk hypotheses with multiple-testing control on calibration data. Model: wraps any fixed predictor. Data/GT: worked computer-vision/tabular examples with task outcomes. Metrics: controlled task risks; numerical experiment results and hardware NOT EXTRACTED. [Primary paper](https://arxiv.org/abs/2110.01052).

Code: license NOT VERIFIED. Weights: not intrinsic to method. Data: example-specific NOT VERIFIED. COMMERCIAL USE NOT CLEARED for imported artifacts. Overlap: calibration-set risk control is prior art. Luma's empirical threshold is not automatically LTT or a statistical guarantee. Repeated regions/images violate naive independent-sample assumptions: use subject-level independent calibration units, specify loss and multiplicity, and do not promise guarantees under arbitrary device shift.

### 13. Suitability of a Mobile Phone Colorimeter Application … Maxillofacial Prosthesis — 2018

Problem/method: smartphone RGB Colorimeter compared with e-skin spectrocolorimeter on skin-colored swatches. Data/GT: swatches and reference instrument. Metrics: repeatability/trueness; numeric results and precise phone/compute hardware NOT EXTRACTED. Authors identify calibration, illumination and distance as remaining issues. [Primary indexed abstract](https://pubmed.ncbi.nlm.nih.gov/30028062/).

Code: proprietary app/no verified grant. Weights: no verified learned component. Data: NOT VERIFIED. COMMERCIAL USE NOT CLEARED for artifacts. Overlap: phone-based skin-color measurement is longstanding. Limitation: swatches do not reproduce living cheek geometry, specularity or physiology; useful instrument protocol precedent, not a ready Luma benchmark.

## Research consequence

Broad novelty claims are untenable: smartphone skin colorimetry, regional CCM, albedo/illumination disambiguation, compact regressors and selective prediction all have close precedents. The narrow remaining hypothesis is whether *measurement-error-supervised regional reliability plus bounded photometric instability* adds useful held-out ΔE00 risk reduction beyond a tuned compact correction/regression/residual-selector baseline. No reviewed source proves this combination absent; patent/FTO search remains separate and NOT STARTED.

Before any publication or external technical novelty statement: extract full quantitative tables/protocols from items 5, 8 and 9, trace their citations and supplements, search patent claims, freeze source versions and reproduce permitted nearest baselines. No external checkpoint or participant dataset was downloaded during this review.
