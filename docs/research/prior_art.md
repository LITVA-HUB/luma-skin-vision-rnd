# Spatial diffusion and alternative statistical representation 2026-09-11

[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382) randomly
bypasses layers during training. Our follow-up always removes the auxiliary
branch at inference and source checkpoint selection; this difference is an
experimental configuration, not evidence of novelty. No external code/weights.

[Diffusion Improves Graph Learning](https://arxiv.org/abs/1911.05485) and
[Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) establish relevant
diffusion and repeated/fixed-point computation prior art. Our positive anchored
Jacobi layer is a finite explicit latent quadratic solve, not a DEQ reproduction,
implicit-gradient method, new physical skin model, or inherently novel architecture.
No paper code or weights were adopted. The54-fit skin source experiment uses
ordinary convolutional controls with similar active capacity and recurrence.

A different candidate representation separates marginal channel distributions
from their dependence structure. [Li et al., Pattern Recognition2017](https://doi.org/10.1016/j.patcog.2016.10.030)
already uses copulas for color-texture dependence; copulas in image analysis
are not new. A possible Luma-specific test would retain absolute RGB statistics
and add rank-dependence context, testing whether that helps native skin-color
error under capture changes. No such trained-model gain is yet established.
Independent monotonic channel transforms preserve ranks, whereas channel mixing,
clipping and spatially varying processing need not; universal camera invariance
does not follow from that mathematical invariance.

# Skin spectral priors: fresh opposing evidence 2026-09-11

[Zhou et al., SCR-AWB, Technologies2025](https://doi.org/10.3390/technologies13060232)
uses skin reflectance priors, device spectral sensitivities and an illuminant
basis to estimate illumination from linear RGB. Its authors explicitly discuss
individual-tone errors from averaged reflectance, and inapplicability to nonlinear
RGB without restoring the assumed relationship. Thus skin spectra plus a small
physical solve is existing research, and mean skin is not a valid universal
color target for Luma. No code/weights/data from that paper were used in a model.
Original author-repository PDF inspected, including equations onp5 and limitations
onpp15–16; CC BY4.0 article, SHA256
`aa9166167a56c3dbd95269e6f4f2a5d92e99bb6b0b947e5a96b8fc48e0312486`.
[Author repository](https://eprints.whiterose.ac.uk/id/eprint/227555/1/technologies-13-00232.pdf).

[Gomes et al. UMINHO paper](https://doi.org/10.1177/00037028241279323)
describes29participants and local spectral variation, including comparison with
other measurement geometries. Our manual source regions do not reproduce those
instrument comparisons. The new spectral PCA and smooth-illumination probes are
ordinary analysis tools and are not a novelty claim.

# Perceptual objective and measured spectra check 2026-09-11

[Sharma, Wu and Dalal](https://doi.org/10.1002/col.20070) provide supplemental
CIEDE2000 test pairs and document formula discontinuities. The new autograd
loss passes all34fixtures plus finite-difference and neutral tests; it does not
make CIEDE2000 globally smooth or introduce a new color metric. Its54-fit
factorial does not support replacing the standard objective wholesale.

The original [UMINHO-HSFD collection](https://doi.org/10.6084/m9.figshare.c.7163569)
provides measured facial reflectance over33bands, and rendered/redacted RGB
derived from those cubes. Original API items and README grant CC BY4.0.
This is a now-verified physical-data route, not a new phone RGB benchmark.
The authors' stated1.3DeltaE00 system error is published evidence, not our
reproduction or a guaranteed measurement floor. [Source audit](../data/uminho_hsfd_verified_inventory.md).

# Fresh paired-invariance mechanism pass 2026-09-11

[VICReg](https://arxiv.org/abs/2105.04906) regularizes agreement, feature variance
and covariance. Our paired-skin adaptation uses true repeated captures and
supervised instrument Lab, not a reproduction of the paper's benchmark.
[Slow feature analysis](https://pubmed.ncbi.nlm.nih.gov/11936959/) is prior art
for learning invariances from changing observations.
[Orthogonal signal correction](https://www.sciencedirect.com/science/article/pii/S0169743998001099)
is spectroscopy prior art for removing unwanted variation; the new hard
within-site covariance projection is a different empirical implementation,
not a new principle or faithful OSC reproduction.

[LeJEPA](https://arxiv.org/abs/2511.08544), November2025, studies joint embedding
and Gaussian regularization. [Weak-SIGReg](https://arxiv.org/abs/2603.05924),
March2026/ICLR workshop per author record, adapts covariance regularization
to supervised optimization. Their large/general-dataset claims do not prove
skin-color accuracy. No code or weights from either was adopted, and the
current experiment does not implement SIGReg. We test a small explicit
VICReg objective and retain its negative direct-color results.

[27local source fits](../benchmarks/skin_pair_v1/report.md): capture consistency
is insufficient for accurate color or successful camera-held-out fitting.
No special-mechanism novelty is established by this pass.

# Historical pixel research update2026-09-11

Pixel mechanism provenance: [Deep Sets](https://arxiv.org/abs/1703.06114) establishes permutation-invariant set learning; [Huber1964](https://doi.org/10.1214/aoms/1177703732) is foundational robust location estimation. Current patch-vote experiments use these established ideas and confidence weighting; no novelty claimed. Direct instrument Lab estimation already appears in the2026skin prior art below. Our weights are trained locally from scratch on original CC-BY data.

# Skin-specific prior art update —2026-09-11

[Bencevic et al.2026](https://arxiv.org/abs/2602.10265) trains EfficientNet-B0 to predict instrument Lab for ITA on MSKCC with patient-level evaluation and extensive additional pretraining. Our direct-Lab objective is therefore not a novelty claim. No pretrained weights or additional training datasets from that work are adopted; its numbers are PUBLISHED BY AUTHORS, not our reproduction. [Weir et al.2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12749783/) provides the original instrument-paired study. Our summary pilot uses a different patient partition and explicitly retains clinical/dermoscopic strata.

# Public color-constancy revision — 2026-09-10

## V4 extension: modern training and repeated correction

The [new synthesis](cc_v4_universal_method.md) compares DINOv2/v3, SigLIP2, V-JEPA2.1, PE, MobileCLIP2, efficient backbones, recurrent-depth/TRM and Self-Refine principles, then translates them into a compact correction-conditioned evidence-routing hypothesis. [Visual-source ledger](cc_v4_visual_training_sources.md), [efficient-source ledger](cc_v4_efficient_training_sources.md), [photometric-source ledger](cc_v4_photometric_sources.md). Multi-Hypothesis2020, CLCC2021, SAFE2026, VLM-CC2026, TPAMI2025 hybrid distillation and Sobolev2017 are close antecedents. Ordinary log transport is exactly an MLP input reparameterization; the implemented nonlinear corrected-simplex transform must beat a generic action-conditioned control. Iteration is established prior art and requires equal-query controls before attributing gains to sequential feedback. No novelty, universality or patent conclusion is established.

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

## Focused update: compact teaching and phone references, 2026-09-11

- Zhao, Luo, Shang and Qu, **Device-specific lightweight color constancy via
  knowledge distillation and fuzzy PID-guided training**, SPIE IPIC2026,
  published2026-07-13, [primary abstract](https://doi.org/10.1117/12.3119239).
  The authors combine teacher illumination features, a compact student,
  pseudo-labels and device-specific adaptation. This directly rules out
  claiming that teaching a small color-constancy model from a large model is
  new. Full tables/protocol and artifact licenses were not verified; no reported
  numbers reproduced. The abstract's linked QLUKD/FPID-KDCC repository returned
  HTTP404 during this check. No code or weights adopted.
- **CSNet: A content and structure-aware approach for color constancy**,
  [primary2026 publisher record](https://www.sciencedirect.com/science/article/pii/S1077314226000056).
  The indexed abstract describes semantic-aware weighting for RGB illuminant
  estimation. Full-page access returned403; architecture/compute/results are
  not sufficiently inspected for an exact comparison. Treat content/semantic
  weighting as known prior art, not a novelty claim for a DINO-trained student.
- Jung, **Comparative Study of Multispectral Image-Based Auto White Balance
  With Optimized Conditions**,2025,
  [primary article](https://doi.org/10.1155/jspe/7775119), uses paired Beyond RGB
  images without a chart as inputs and a median gray patch from the chart
  capture as illumination supervision. Its manually filtered population and
  multispectral protocol differ from our paired-phone field screen. No
  author numbers are entered into locally reproduced tables. This supports
  the reference construction but does not validate our specific quality filter,
  camera transfer result or a physical skin-color claim.

The optional Apache2.0 DINOv2 teacher is only acquired and load-checked so far;
it has not supervised a model. Any later raw-versus-canonical teacher experiment
must keep true illumination available only during training, compare ordinary
distillation and no-teacher controls, and keep the photometric input stream
separate from desired semantic invariance. Canonicalizing a training image by
its illuminant is an approximation, not measured intrinsic surface color.
The current priority is finishing paired V5 and equal-query controls and the
phone reference/evaluation pipeline. New teacher results remain NOT MEASURED.

## Research consequence

Broad novelty claims are untenable: smartphone skin colorimetry, regional CCM, albedo/illumination disambiguation, compact regressors and selective prediction all have close precedents. The narrow remaining hypothesis is whether *measurement-error-supervised regional reliability plus bounded photometric instability* adds useful held-out ΔE00 risk reduction beyond a tuned compact correction/regression/residual-selector baseline. No reviewed source proves this combination absent; patent/FTO search remains separate and NOT STARTED.

Before any publication or external technical novelty statement: extract full quantitative tables/protocols from items 5, 8 and 9, trace their citations and supplements, search patent claims, freeze source versions and reproduce permitted nearest baselines. No external checkpoint or participant dataset was downloaded during this review.


## 2026-09-11 phone and broader-search evidence

The first source-only Samsung/Oppo evaluation is measured, including a documented
HDF5 name repair and preserved original results: [report](../benchmarks/phone_v1_alias_report.md).
V2 remains stronger than V5 on phone selective risk; no universal/skin claim.
[Mechanism-search ledger](../research/aggressive_search_2026_09_11.md) and
[V6 execution](../research/cc_v6_execution_status.md) preserve combination failures
and investigate non-CNN Fourier regression and correction decision sets.

## Semantic/sensor and alternative-representation review, 2026-09-11

The earlier teacher-acquisition-only status is superseded: four views per
1126 TRAIN images have now been extracted locally. No held-out teacher features
were extracted. The five-arm V7 source screen is running; no teacher-based
camera-transfer improvement is yet measured.

- [Integral Fast Fourier Color Constancy](https://arxiv.org/html/2502.03494v1),
  Wei et al., CVPR 2025: integral UV histograms, parallel Fourier prediction and
  spatial smoothing already provide efficient regional/multi-illuminant AWB.
  Regional Fourier histograms are not our novelty. The full primary HTML was
  inspected; author size/speed figures are not locally reproduced numbers.
- [GCC](https://arxiv.org/abs/2502.17435), 2025: generative color-checker
  inference is an existing alternative representation; its abstract describes
  deterministic one-step diffusion and cross-camera use. No assets adopted.
- [Deep Image Harmonization with Globally Guided Feature Transformation and
  Relation Distillation](https://openaccess.thecvf.com/content/ICCV2023/papers/Niu_Deep_Image_Harmonization_with_Globally_Guided_Feature_Transformation_and_Relation_ICCV_2023_paper.pdf),
  ICCV 2023: clean-target feature/relation distillation is established adjacent
  prior art. The exact V7 GT-corrected-teacher experiment remains a hypothesis,
  not evidence that canonical supervision is a new general method.
- [NightCC](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_NightCC_Nighttime_Color_Constancy_via_Adaptive_Channel_Masking_CVPR_2024_paper.pdf),
  CVPR 2024, already studies mean-teacher nighttime adaptation and correction
  feedback. V7 uses no test-camera adaptation; teacher use itself is not new.
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814) and
  [Non-Exchangeable Conformal Risk Control](https://arxiv.org/abs/2310.01262)
  make their sampling/shift assumptions essential. Our existing 11 CAL groups
  cannot support a nontrivial standard group-level 95% split-conformal bound
  without additional assumptions: ceil((11+1)*.95)=12 exceeds 11 calibration
  scores. Do not advertise guaranteed unseen-camera rejection from an empirical
  risk head. A correction-set branch remains exploratory.

The restricted sensor transform M=D((1-epsilon)I+epsilon*A), A nonnegative
row-stochastic, is a mathematically admissible combination of source spectral
sensitivities. It spans neither all camera responses nor nonlinear ISPs.
Sensor simulation, semantic color constancy, privileged training targets and
knowledge distillation have established precedents. The useful unresolved
question is whether this particular matched compact training setup improves
real camera transfer and selective error at fixed deployment cost.
# 2026-09-11 direct skin-color endpoint update

[He et al., Development of an image-based measurement system for human facial
skin colour](https://doi.org/10.1002/col.22737) provides direct overlap for
camera-RGB-to-measured-skin-XYZ calibration. Linear, polynomial, root-polynomial
and neural calibration are established methods. The [original CC BY4.0
data](https://zenodo.org/records/5532176) are now used in our bounded author-split
XYZ baseline, with no exact MATLAB reproduction or perceptual-error claim.

[CHROMA-FIT](https://research.fit.edu/idl/publications/),
[DAST](https://github.com/dasec/DAST-SkinTone-database) and
[ENCoDE](https://physionet.org/content/encode-skin-color/1.0.0/) demonstrate
existing instrument-paired skin-tone evaluation directions. Their publication
does not automatically clear their data for local/commercial training.
[Separate access/rights inventory](../data/public_skin_color_inventory.md).

Our completed V7 real-camera result is negative for canonical semantic teacher
targets. It cannot support a new accuracy claim. Future novelty must concern
an experimentally useful mechanism and a valid skin-color endpoint, rather
than renaming calibration, distillation, uncertainty or color invariance.


## Instrument-reference and paired-objective source check (2026-09-11)

36 new source fits do not establish an advantage from emphasizing paired mean
color bias. Increasing consistency slightly helps mixed mean but worsens
selective error and one transfer direction. This is a standard squared-loss
decomposition, not demonstrated invention. Instrument-supervised skin learning
has existing prior art. See [decision](skin_correspondence_next_decision.md) and
[fresh original-method checks](skin_correspondence_sources_2026_09_11.md).


## Frozen teacher skin readout screen (2026-09-11)

108 native-Lab source readout fits do not establish a compact or universal
advantage. Global pooled DINOv2 plus absolute color helps weak linear controls
in two protocols, but loses to strong compact models in all three. Original
Apache 2.0 teacher is 22.06M parameters; no student was trained or deployed.
This is a representation diagnostic, not new invention. Preserve both the
positive weak-control comparison and the negative strongest-control results.
[Evidence and next local-correspondence falsifier](skin_teacher_next_decision.md).


## Local descriptor correspondence and physical-prior redirection (2026-09-11)

36 source fits reject this aligned-teacher adapter as an accuracy improvement;
global/shuffled descriptors do not establish a local-correspondence advantage.
Post-hoc gradients do not prove capture-mode supervision caused the harm.
No architecture novelty or compact-student accuracy is claimed. Original ISSA
v4 CC BY4.0 rights and acquisition now verified; measured reflectance priors
are the next different source of knowledge, with colorimetry/support checks
required first. No spectral or photo accuracy from ISSA is measured yet.
[Decision and evidence](skin_local_teacher_next_decision.md).

## ISSA material screen and recent optical-locus prior art

[The optical origin of the human skin color banana in CIELAB space](https://pmc.ncbi.nlm.nih.gov/articles/PMC13307969/) links physical skin models and ISSA color geometry. Skin-material manifolds, chromophore models and spectral PCA are not novel. Our fixed controls show representation fidelity, not image inversion. See [material report](../benchmarks/skin_issa_v1/report.md).


## Conditional native-skin-color distribution screen (2026-09-11)

The 45 measured-material image fits do not establish a nonlinear-prior advantage;
see [decision](skin_material_next_decision.md). Next cancel the single-point
prediction assumption and test distributional predictions with expected-DeltaE00
inference. This is an unverified combination, not a novelty claim.

Fresh primary sources: [Bishop 1994, Aston](https://research.aston.ac.uk/en/publications/mixture-density-networks/)
establishes neural conditional mixture distributions, including inverse problems.
[Conditional risk minimization](https://arxiv.org/abs/1611.07096) establishes
prediction by minimizing an estimated loss-dependent conditional risk.
[Yang et al. 2026](https://arxiv.org/abs/2602.19055) studies disentangled captured
skin appearance and counterfactual editing for lesion images; its abstract
reports augmentation/classification utility, not our instrument-Lab endpoint.
No code, weights or datasets from these sources are adopted. The Aston paper
itself has CC BY-NC-ND terms; reading prior art does not clear third-party code
or weights. Search is a scoped update, not an exhaustive patentability review.


## Measured conditional-color density update (2026-09-11)

36frozen compact fits complete. Four-component density does not beat the strong
ordinary image regressor consistently. A single Gaussian improves mixed-source
risk ranking at80% (3.2709 versus3.4407DeltaE00) but loses that advantage in both
camera-transfer protocols. All source data is repeatedly inspected; no new
independent confirmation. [Decision](skin_distribution_next_decision.md).
A targeted post-hoc quadrature diagnostic separates numerical approximation
from model failure. Next hypotheses: matched color/risk crossing, and learning
conditional downstream-loss surfaces without reconstructing a full density.
Established MDNs and conditional risk minimization are prior art; no novelty
or calibrated-selectivity claim follows from this implementation.


## Native skin risk crossing and candidate-loss fields (2026-09-11)

Ninety fixed crossing endpoints and36matched loss-field fits are complete.
Mixed80%crossing3.1954 versus matched ordinary pair3.3625 has a descriptive
patient interval crossing zero and worsens reverse transfer. Direct candidate
loss learning, including removal of probability nonnegativity, does not yield
universal accuracy. [Decision](skin_risk_field_next_decision.md).

Fresh primary-source review: [Imani et al., JMLR2026](https://jmlr.org/papers/v27/24-0260.html)
finds histogram regression gains can arise from optimization, not extra target
information. Our soft categorical control is related, not a reproduction of
all their protocols. [Dheur/Ben Taieb2024](https://proceedings.mlr.press/v238/dheur24a.html)
studies training-integrated quantile recalibration; uncalibrated Gaussian output
must not be represented as guaranteed error. No code/weights adopted.
The current native-Lab loss field also relates to this repository's earlier
illuminant action-field work and established conditional risk minimization.
Changing the endpoint to skin color does not by itself establish novelty.
