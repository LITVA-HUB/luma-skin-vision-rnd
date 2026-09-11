# Spatial/training-only branch decision 2026-09-11

90real-photo source fits do not establish a better universal architecture.
Spatial graph inference fails the mixed-source controls. Training with a graph
branch but removing it at inference has a reproducible one-direction source
transfer benefit (4.9736 versus5.5089, all3seeds), with mixed/reverse losses.
This is an interesting mechanism to isolate, not patent novelty or independent
validation. A fixed ordinary two-model combination still wins mixed-source
accuracy; proposed combinations do not win across protocols or selective risk.

Training-time nuisance perturbation with an unchanged inference core is the
next bounded hypothesis. It must outperform ordinary perturbation/DropPath and
constant-calibration controls while keeping absolute color and acceptable tails.
Graph layers, branch dropping and ensembles alone have established prior art.
Retain the marginal/dependence alternative rather than narrowing all research
to this one favorable transfer direction. [Evidence and next falsifiers](skin_spatial_next_decision.md).

# Material/spatial hypothesis after real-spectral falsifiers 2026-09-11

Skin spectral priors, PCA, positive illuminant models and physical inversion
are not new in themselves. SCR-AWB2025 is directly relevant prior art; its
averaged reflectance and sensor knowledge do not satisfy our unknown-person,
unknown-camera skin measurement objective. Three-face source probes support
a compact material family but do not demonstrate RGB inversion or an exact
illumination collision. Shared spatial constraints may help; their survival
through RGB integration and a nonlinear ISP remains UNVERIFIED.

Next candidate: shared capture reasoning from spatial skin-color relations,
retaining local shading and absolute nativeLab color, with uncertainty tied to
remaining ambiguity. A special solver must beat both the plain patch system
and a matched ordinary spatial network. Merely adding a graph, skin prior,
recurrence or confidence output is not a defensible novelty claim. No observed
new real-photo gain yet. [Decision and failure modes](skin_spectral_next_decision.md).

# Capture-conditioning decision 2026-09-11

The54-fit source factorial shows a directional gain, not a universal mechanism
win: MSEmixture SLR->iPod5.0193 beats matched auxiliary control5.7385 and plain
5.8301 across all3seeds. Reverse mixture5.9635 is worse than plain5.5089 across
all3seeds. A matched auxiliary control alone can be weak because its additional
classification objective harms useful representation. Always retain the plain
control. Direct squaredCIEDE2000 fails to improve source averages and harms
reverse-camera transfer. [Full evidence](../benchmarks/skin_capture_v1/report.md).

Conditional experts, mode classification and a perceptual loss are existing
components; no patent novelty claim follows. The next fundamentally different
candidate is a physically constrained skin-material representation using actual
measured face spectra. Original UMINHO-HSFD CC BY4.0 now verified. First verify
masking and reference conventions, and test ambiguity of color reconstruction
under plausible acquisition changes. Rendering these spectra can provide
derived inputs for falsifiers, not a substitute for real-phone skin validation.

# Paired-capture invariance decision 2026-09-11

New source experiments reject unconditional capture invariance as the current
route to a large skin-color gain. Output consistency and VICReg improve
same-site repeatability but not direct instrument accuracy; hard nuisance
projection worsens accuracy. Output consistency loses all six source-camera
held-out seed/direction contrasts. These27fits and81replays are negative or
diagnostic evidence, not a new innovation claim. [Report](../benchmarks/skin_pair_v1/report.md).

Next candidate inverts the assumption: estimate latent capture process from
one image and condition the color reconstruction, preserving useful nuisance
information rather than removing it. Compare against equal-capacity models
receiving identical auxiliary capture-mode supervision. Mixtures and auxiliary
tasks are not novel themselves. Any narrow contribution must demonstrate
actual direct-color/coverage/compute advantages and independent replication.
Separately examine perceptual-loss geometry as a standard objective control.

# Independent direct-skin test decision 2026-09-11

Primary matched-search selective hypothesis: small observed80%DeltaE00 gain
4.333 to4.159,95%patient-cluster interval crosses zero. Identical color models;
different source-selected head families (MLP/HGB), so feature causality is not
isolated. Ordinary6-model fusion+density reaches4.145. No convincing novel
mechanism win; no product-level facial-phone accuracy established. Current
test is exposed and must not be recycled as fresh confirmation. Prioritize
source-only direct-color/within-site consistency hypotheses with a separate
future confirmation cohort. [Independent evidence](../benchmarks/skin_mskcc_selective_v1/report.md).

# Historical pixel research update2026-09-11

Direct pixel evidence: patch-set color regression is promising on source validation, but standard set aggregation is not new. Iterative Huber mechanisms failed to reliably improve matched pooling; forcing local-only votes lost useful context. Next investigate matched, calibrated expected skin DeltaE00 rather than claim a new architecture from a baseline win. See [current decision](skin_mskcc_pixel_decision.md).

# Direct skin endpoint update —2026-09-11

Direct native-Lab regression is not novel: current MSKCC summary MLP is a baseline. Frozen source validation meanDeltaE00 4.3702; density rejection failed. Next falsifier removes obligatory illuminant estimation and tests locally extracted color/texture evidence and training-only paired acquisition consistency. No novel mechanism or phone accuracy established. See [decision](skin_mskcc_next_decision.md).

# Active public-benchmark hypothesis

2026-09-11 latest decision: V7 canonical semantic targets failed the frozen
real-camera comparison, including the equal-budget raw-teacher C+. No proposed
V7 model beats the strongest observed Fourier control. [Complete result](../benchmarks/cc_v7_external/report.md).
Preserve limited benefits of sensor augmentation as a known training mechanism,
not a new validated contribution. The basis-free projector probe is numerically
sound but its positive-weight hypothesis is restrictive; signed geometry is
unverified. Neither projector algebra nor distillation is novel by itself.

The user now prioritizes actual skin-color accuracy as the main endpoint.
[Instrument-paired pilot](../benchmarks/skin_he_xyz_v1/report.md) measures
regional RGB-to-XYZ calibration against real skin readings, with explicit
white-reference and camera limits. It is ordinary calibration, not a proposed
new architecture. Next contribution must improve a valid skin-color/reliability
endpoint, not only illuminant angular error. [Product target](skin_color_target_2026_09_11.md).

Earlier hypothesis states below are historical and do not supersede this result.

**V5 final update:** all18 runs finished. Generic action has lower mean error
(2.4136° versus transport2.4548°), while transport has lower mean raw risk80
(2.0192° versus2.1333°). Selected-action/derivative training does not improve
consistently. Equal-query control across all three seeds shows no useful
feedback-specific advantage. [Final evidence and next decision](../benchmarks/cc_v5_report.md).
Retain the modest risk-ranking hypothesis, change the failed training direction,
and first test frozen models on the acquired phone data. No new validated
innovation or universal model is claimed by this source-development screen.

## V5 status and narrowed next question

The paired-state experiment now has two completed seeds, six arms each;
seed43 remains running. [Measured interim results](../benchmarks/cc_v5/two_seed_screen/report.md)
and all24 CPU checkpoint replays are preserved. Selected-action supervision and
physical derivative matching do not consistently beat ordinary random-action
training. The best arm changes by seed. Do not infer a robust new contribution.

The [equal-query control](../benchmarks/cc_v5/two_seed_search_control/report.md)
further weakens recurrence as the present contribution: all10 trained critics
retain the original point at the coarse stage on all119 validation images;
adaptive and fixed51-query answers are identical. At103 queries the mean-error
differences are small and mixed. A declining predicted cost alone is not
evidence of successful self-correction. Preserve this negative and finish the
third seed before deciding which estimator/critic to carry forward.

The next defensible question is whether stronger training-time scene context
improves a compact physical estimator and its risk prediction without a large
inference model. Semantic weighting and teacher distillation already exist in
2026 color-constancy work; [focused prior-art update](prior_art.md). Any raw or
GT-canonicalized teacher must be compared with ordinary distillation and a
matched no-teacher control. GT canonicalization is training-only privileged
information and does not create physical surface-color measurements. No new
teacher architecture or accuracy result is claimed yet.

Phone generalization is now a concrete evaluation axis: Samsung/Oppo data is
being acquired, a fixed reference policy was derived from three TRAIN scenes,
and an auxiliary iPhone repeatability archive is acquired. No held-out phone
accuracy, generic JPEG/HEIC correction or facial measurement is established.
An eventual contribution must survive these external tests and the strongest
matched C+, beyond source-development gains. The positive V2 result remains
the strongest completed evidence; it is not a universal or patent claim.

## V4 investigated mechanism

Latest user steering broadens intended generalization across scenes, surfaces, illumination and processing, with camera as one axis. The [selected mechanism](cc_v4_universal_method.md) is compact correction-conditioned routing of cached local evidence, nonlinear physical correction of color statistics, exact reproduction-cost supervision and1/2/4-stage refinement. It is implemented as a3.097M network with matched posterior and generic action controls; see [specification](cc_v4_spec.md) and [source-development protocol](cc_v4_source_lock.md). The initial linear transport idea was rejected at preflight because a first MLP layer absorbs it exactly. The nonlinear transform remains an inductive-bias hypothesis, not a new information source. Teacher transfer, EMA, Sobolev supervision, candidate scoring and iterative refinement are established prior art. Actual results belong in the separate [V4 report](../benchmarks/cc_v4_report.md); no universal or facial-color claim follows. V3's planned graph revision below is preserved historical direction, superseded as first priority by this experiment.

## V3 result and revised direction

**INITIAL FULL-FRAME HYPOTHESIS FAILED ITS SOURCE SCREEN.** An image-derived color basis, graph diffusion and transported directional posterior were implemented and trained at1.216M parameters. Full-frame mean/risk80=4.281°/4.155° versus matched direct2.547°/2.304° on119 reused development-validation images. This establishes neither a new positive benchmark result nor novelty. GL frame/canonicalization has published antecedents. A graph retaining a separate global color state, with learned fusion and camera-space risk supervision, is the next **PLANNED, UNVERIFIED** candidate. See [decision](cc_v3_revision_decision.md), [measurements](../benchmarks/cc_v3_report.md) and [prior art](cc_v3_prior_art.md). Old negative results and V2's bounded positive camera-transfer result remain intact.

See [ranked public decision](public_hypothesis_decision.md), [fresh prior art](public_color_prior_art.md) and [frozen experimental protocol](public_protocol_v1.md). Compact single-image transfer and selective reproduction-risk estimation are UNVERIFIED research hypotheses; algorithm mixtures, error prediction and abstention are not independently novel. No physical ΔE00 or facial-color claim follows from illuminant ground truth. Historical facial hypothesis below is deferred while proprietary collection is unavailable.

## V2 investigated hypothesis — bounded empirical support

**V2 MEASURED; NO NOVELTY CLAIM.** Fresh384-image risk80 improves5.558°→3.805° against the strongest matched direct C+; combined risk improves4.266°→3.805° against context-only risk on the same residual model. Source/Canon regress, cheap GW+ridge remains statistically competitive at80%, and exploratory GW residual has lower full mean. See [full effect and limits](../benchmarks/cc_v2_report.md). Prioritize C with the measured B ablation; A remains reproduction-angular proxy supervision, not physical surface color-error supervision. Test whether an anchor-normalized compact residual estimator and a separately fitted risk head transfer better than a matched direct estimator and its strongest feasible C+ selector. For a channelwise positive-diagonal-equivariant anchor `a`, the investigated estimator is

```
e(x) = normalize(a(x) * exp(r(x / a(x)))).
```

This wrapper is known: [Cotogni and Cusano 2022, Eq. (19)](https://arxiv.org/pdf/2207.00292) already gives normalization, arbitrary prediction and restoration; their [2024 illuminant-equivariant paper](https://link.springer.com/chapter/10.1007/978-3-031-72845-7_18) supplies additional direct task overlap. The [v2 prior-art analysis](cc_v2_prior_art.md) documents exact assumptions, recent overlap, access limitations and strong baselines. Mathematical gain equivariance is an implementation property, not evidence of new-camera accuracy or calibrated uncertainty.

Preserve the candidate meanings: **A** is downstream-color-risk selective normalization, currently assessed only through reproduction angular risk; **B** is cheap hypotheses/disagreement plus contextual residual prediction; **C** is compact single-image unseen-camera generalization. V2 investigates B within C, with A as the selection objective. **C+ remains the strong matched control**, never the proposed method.

The empirical question is whether invariant context and relative candidate log-ratios improve selective reproduction risk at comparable coverage and cost beyond cheap anchors, fixed source correction, matched direct/residual models, and simple uncertainty scores. Recovery error is reported separately because it is not diagonal-gain invariant. Fit the estimator on source training, choose hyperparameters/checkpoints on source validation, fit the risk head on disjoint source residual groups and calibrate on a separate source split. Freeze all choices before fresh target evaluation; use no target-camera batch statistics, calibration matrix or target labels during fitting. Match augmentation, capacity, seeds and budget; preserve unhelpful variants. See the [v2 experimental plan](cc_v2_plan.md).

The [v1 mixture decision and negative results](public_hypothesis_decision.md) remain unchanged, including failure to beat the strongest selective control and loss to Gray World on Sony. Official462 and Sony30 are previously observed regression sets, not fresh discovery evidence; official split scene/date overlap remains a limitation. Passing equivariance tests or improving a source metric cannot override those findings. The investigated effect must survive locked camera evaluation and strong baselines before any contribution is proposed; it would still not establish patentability or physical facial ΔE00 accuracy.

---

Historical synthetic/facial-stage material follows.

# Novelty hypothesis — provisional, 2026-09-10

Status: NOT VALIDATED. Literature analysis is evidence about prior art, not a scientific result or patentability opinion. See [prior art](prior_art.md).

Investigate a narrow mechanism: estimate how instrument-referenced cheek color changes across bounded corrections and ROI perturbations; combine this instability with region suitability supervised by subject-held-out colorimetric error; learn an error ranking and calibrate an accept/retake policy on separate subjects. The target is CIELAB D65/2° under the documented instrument protocol within an explicit operating domain.

The potential effect is lower mean and tail ΔE00 at the same accepted-image coverage than A0/A1/A2, a compact regressor C and a matched strong C+ selector. C+ must have the same backbone, training subjects, metadata access, optimization budget and ordinary quality/error features. The proposed method only earns a contribution if its particular ambiguity/ROI information improves over C+, including on held-out camera or lighting domains. Merely giving the proposed model more parameters or supervision does not answer the question.

Known mechanisms are not ours: DeepWB's WB alternatives; C5's cross-camera adaptation; TRUST's scene disambiguation; regional CCM in [Kang and Kim v4](https://arxiv.org/abs/2512.21988v4); SelectiveNet's learned rejection; LTT's calibrated risk control. Instrumented smartphone skin-color work further removes a broad first-of-kind claim. The incomplete 2026 YLGTD extraction is a material novelty uncertainty, not evidence of a gap.

## Falsification and minimum evidence

1. Demonstrate instrument repeatability before model fitting; fix registration and acquisition if its error is comparable to the proposed gain.
2. Lock participant splits and deployment tolerance before final testing. Count both cheeks/repeats as correlated observations, bootstrap subjects.
3. Tune classical global and regional device CCM baselines on training only; reserve a separate applicability mode when a device profile is needed.
4. Fit error models to out-of-fold or truly held-out subject residuals. Calibrate once on disjoint subjects. Report reliability of the threshold event and tail error, not just error-score correlation.
5. Ablate ROI reliability, hypotheses, ambiguity, error head; include no-correction, conventional selector, matched C+, and camera/lighting holdouts. Freeze useful coverage levels before reading test results.
6. Reject the innovation claim if the matched advantage is absent, inconsistent across seeds/domains, or explained by leakage/extra budget. Publish negative findings internally; choose a simpler calibrated-device or guided-capture pivot only if evidence supports it.

No learned measurement-aware ROI is validated yet. Heuristic masks, synthetic reliability labels and randomly initialized smoke models are engineering scaffolding. A finite-sample risk guarantee requires an implemented statistical procedure and its assumptions; empirical calibration alone is not that guarantee. Compression and deployment do not establish novelty. All real effect sizes and confidence intervals: NOT MEASURED.


## 2026-09-11 phone and broader-search evidence

The first source-only Samsung/Oppo evaluation is measured, including a documented
HDF5 name repair and preserved original results: [report](../benchmarks/phone_v1_alias_report.md).
V2 remains stronger than V5 on phone selective risk; no universal/skin claim.
[Mechanism-search ledger](../research/aggressive_search_2026_09_11.md) and
[V6 execution](../research/cc_v6_execution_status.md) preserve combination failures
and investigate non-CNN Fourier regression and correction decision sets.

## Current hypothesis decision, 2026-09-11

[V6 is complete and negative](../benchmarks/cc_v6_report.md): the canonical-frame
combination does not improve source accuracy over V5, and no V6 real-camera
advantage is measured. V5's phone failure and V6's source failure are retained.

V7 challenges a different assumption: the deployment architecture may not need
additional iterative modules if training can transfer useful scene structure
from a GT-corrected, training-only semantic teacher. Its strongest C+ has the
same teacher, student, projection, sensor augmentation, data and training budget,
but uses raw teacher views. A canonical/raw target difference is the candidate
mechanism. No novelty or accuracy advantage is established by implementing it.

The deployment estimator has 3,033,651 parameters; training adds 369,024
projection parameters, removed at inference. [Training protocol](cc_v7_semantic_sensor_protocol.md)
and [transfer protocol](cc_v7_transfer_diagnostics_protocol.md) separate source
development, virtual sensor diagnostics and later fresh real-camera evaluation.
Complete all five arms and three seeds, retain failures, fit matched risk heads
only on held-out source residuals, and freeze every method before external
decoding. A benefit must survive the raw-teacher C+ and no-teacher controls.

Invariance is desired for scene features; the camera-RGB illuminant must remain
equivariant to the sensor. Exact full color-frame equivariance was already
implemented and failed in V3, so algebra alone does not justify repeating it.
Single-image ambiguity still requires rejection. Neither DINO pretraining nor
virtual RGB mixing supplies genuine corresponding surface-Lab ground truth.


## Skin correspondence and paired-objective falsification (2026-09-11)

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

## Measured-material branch: not yet a photo-accuracy contribution

Fifteen fixed ISSA oracle controls demonstrate that2latent variables lose color and8can preserve it well with full spectral input. This does not solve the inverse problem. An image material decoder must beat matched ordinary bottlenecks; a competing feasible-color-set branch should refuse when plausible explanations disagree. Observer/support mismatch and duplicate source records are explicit limits. [Decision](skin_issa_next_decision.md).


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


## Observed skin patch support and routing (2026-09-11)

45 matched real-photograph fits show only a directional source transfer gain,
with mixed-population harm. True local source labels do not rescue frozen
experts on TRAIN virtual bags. Neither patch augmentation nor mixture routing
is claimed as new. [Decision](skin_capture_support_next_decision.md).
Prior art includes [AugMix](https://arxiv.org/abs/1912.02781),
[MixStyle](https://arxiv.org/abs/2104.02008),
[StyleMix](https://openaccess.thecvf.com/content/CVPR2021/html/Hong_StyleMix_Separating_Content_and_Style_for_Enhanced_Data_Augmentation_CVPR_2021_paper.html)
and [MixUp-MIL](https://arxiv.org/abs/2211.05862). These are related augmentation
mechanisms, not protocols reproduced here. Generic color/style invariance may
remove the desired skin color information. Restricting derived bags to observed
same-site patches preserves reference provenance, but proves neither physical
realizability of the composed image nor technical novelty. No external code
or weights from these works adopted. No skin accuracy gain follows from an
illuminant-angle result.


## Expert removal and conditional skin supervision: source review (2026-09-11)

The new screen contrasts removal of the mixture against uniformly and
capture-conditionally supervised native-Lab heads; coefficients and 72 fits
were frozen before training. This is a mechanistic ablation, not novelty.
[Makkuva et al., ICML 2019](https://proceedings.mlr.press/v97/makkuva19a.html)
already study joint expert/gate learning difficulties and a separate expert
estimation approach using cross-moments. Their theorem is not transferred to
our nonlinear skin regressor, and their algorithm is not reproduced here.
No external code or weights adopted.

HUST (ICCV 2025) remains relevant facial-albedo prior art already inventoried.
A fresh CVF full-text fetch failed; no new table values or rights inferred.
The [DAST original repository](https://github.com/dasec/DAST-SkinTone-database)
still directs users to contact its authors for the full dataset and does not
establish a permissive dataset license on its visible page. No contact or
sample acquisition occurred. The [He2021 original deposit](https://zenodo.org/records/5532176)
still provides paired regional RGB/XYZ tables, not additional full photographs
or a newly verified numeric reference white. No replacement DeltaE00 targets
were manufactured. These checks do not clear a new ordinary-phone dataset.


## Conditional skin appearance inversion (2026-09-11)

108 source fits and separate empirical-support controls do not establish an
innovation win. Mixed inverse3.8663 versus direct3.9046 has a patient interval
crossing zero; matched hull-constrained direct3.8947 narrows the difference.
[Decision](skin_appearance_inverse_next_decision.md). Gaussian mixture regression
is established; see [Calinon et al.2007, author page](https://calinon.ch/paper4003.htm).
Bayesian inversion for color already has a substantial history, including
[non-Gaussian color constancy, NeurIPS2003](https://papers.neurips.cc/paper_files/paper/2003/hash/c65d7bd70fe3e5e3a2f3de681edc193d-Abstract.html)
and [multi-hypothesis color constancy, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Hernandez-Juarez_A_Multi-Hypothesis_Approach_to_Color_Constancy_CVPR_2020_paper.html).
These are conceptual prior art, not locally reproduced author protocols. Our
endpoint is actual skin native Lab, but changing the endpoint does not establish
novelty. No external code or weights adopted. A fixed-covariance Gaussian patch
model has a sufficient-mean limitation; simply processing every patch must not
be called a new source of color information without testing that limitation.


## Graph/support and patch-distribution falsifiers (2026-09-11)

84 additional source fits use actual instrument skin Lab / DeltaE00. A graph
restricted to original training grids plus observed-patch support improves new
transfer controls, but does not beat strongest historical models. A three-mode
conditional patch distribution beats one Gaussian, yet mixed mean-trained
control4.0120 is better than full-patch4.0554. All 12 single-Gaussian fits satisfy
the sufficient-mean identity. All 48 fits choose strength1 from 1/4/16/64;
this does not establish the globally optimal temperature. Existing graph,
mixture and Bayesian mechanisms remain prior art; no new invention is asserted.
[Decision, assumptions, failure modes and next falsifier](skin_graph_patch_next_decision.md).


## Support / representation decision

The 27-fit TRAIN-only screen does not establish a new representation advantage.
An active learned pixel residual barely changes when patch texture is removed.
Investigate TRAIN-color support allocation under fixed data/update controls next;
sampling balance is established methodology, not the candidate invention itself.
[Mechanism, failure mode and next falsifier](skin_support_curve_next_decision.md).


## Color allocation: partial result only

Color balancing6.0012 versus person/site6.0084 is not a convincing mechanism
advantage. The within-person permutation control gives a limited signal worth
a wider matched source test. Combine uniform person mass with within-person
color emphasis, without asserting novelty for balancing or importance sampling.
[Assumptions, controls and next decision](skin_color_sampling_next_decision.md).


## Combination tested across source acquisition (2026-09-11)

54 matched930-step skin fits reject the universal-transfer claim for person/color
sampling. Mixed3.7435 versus color3.7864 is inconclusive; unseen directions lose
stronger controls. Balancing the marginal target distribution does not establish
an acquisition-invariant conditional appearance mapping. This is a local result,
not a theorem that all camera-blind models fail. No new novelty is claimed for
hierarchical sampling, density weighting or importance correction.
[Decision and next falsifier](skin_sampling_transfer_next_decision.md).


## Offset diagnostic and next hypothesis

Shared correction inferred from other people's instrument references is not
sufficient across source transfer directions. It is a privileged comparator,
not improved product accuracy. Test whether person-specific training updates
transfer across groups, with random-group and finite-step controls before any
meta-learning fit. MLDG/Fish/Fishr are prior art; no gradient novelty is asserted.
[Evidence and decision](skin_offset_diagnostic_next_decision.md).

## Gradient falsifier: insufficient grounds for invariance (2026-09-11)

The controlled TRAIN audit does not support universal harmful person-gradient
conflict: within single-camera models the conflict also occurs under random
grouping. A successful local update can worsen actual pooled skin DeltaE00.
Do not claim gradient matching as our novelty or optimize agreement as a proxy
for skin color. No new independent model victory occurred.

Next investigate relational color compatibility as an unverified alternative,
starting with the additive/cycle-consistent degeneracy: f(x)-f(anchor) plus
reference anchoring is merely an absolute predictor plus an offset. Learned
pair-context quality must beat this and ordinary regression, with no inference
camera labels, no held-person encoder leakage and all support memory counted.
Known metric/relation learning prior art must be reviewed before novelty claims.
[Decision and cheapest falsifier](skin_gradient_transfer_next_decision.md).

## Relational probe: matching is insufficient (2026-09-11)

No unique comparative mechanism is established by same-site discrimination.
Simple excluded-person ridge36 has70.08% nearest-control preference versus
descriptive learned context70.92%, and actual skin mean5.4259 DeltaE00. Pair
recognition is not percentage color accuracy. Additive reference comparisons
reduce to an absolute predictor plus a constant offset; cycle consistency
alone does not add acquisition information. Known relations/kernels/Hodge
methods remain prior art, not project novelty.

The next narrow test is query-dependent support weighting on direct skin Lab,
with standard local-affine and weighted-mean controls before a learned pair
mechanism. No exposed independent TEST/CAL tuning, new phone proof, or claim
of an already improved compact architecture.
[Decision](skin_relational_probe_next_decision.md).

## Local-affine component survives; universal claim fails (2026-09-11)

On24 excluded-person TRAIN folds, color-weighted affine regression reduces
actual mean skin DeltaE00 by15.48% versus global ridge and improves21 people.
Color-weighted reference averaging alone loses to ridge. This favors a local
mapping component rather than merely copying nearby reference colors.
Both unseen-camera directions still lose to strong historical neural controls,
and nearest-support rejection can raise accepted-image error. These exploratory
results neither prove a novel method nor change independent-test accuracy.

Next unverified hypothesis: a strong compact learned representation may supply
better neighborhoods for reference-conditioned residual correction. Test it
against the unchanged base and ordinary residual C+ at matched capacity and
budget; no inference camera ID or query reference. Authorized total neural cap
is1,129,297, including the929,297 base. A193,795-parameter candidate adapter is
budgeted but not implemented/trained. Store and report all non-neural payload.
Begin with one correction pass; additional passes require separate evidence.
[Decision](skin_local_reference_next_decision.md) and
[capacity contract](skin_capacity_budget_2026_09_11.md).
