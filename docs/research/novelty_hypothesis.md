# Active public-benchmark hypothesis

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
