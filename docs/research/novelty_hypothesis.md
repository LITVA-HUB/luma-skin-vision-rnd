# Active public-benchmark hypothesis

See [ranked public decision](public_hypothesis_decision.md), [fresh prior art](public_color_prior_art.md) and [frozen experimental protocol](public_protocol_v1.md). Compact single-image transfer and selective reproduction-risk estimation are UNVERIFIED research hypotheses; algorithm mixtures, error prediction and abstention are not independently novel. No physical ΔE00 or facial-color claim follows from illuminant ground truth. Historical facial hypothesis below is deferred while proprietary collection is unavailable.

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
