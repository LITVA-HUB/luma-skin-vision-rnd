# Mechanism search ledger: challenge the representation

Latest completed tests: V7 lost its real external canonical-teacher contrast;
Fourier ridge outperformed all V7 families on pooled full/risk80. The projector
probe is now MEASURED: all numerical invariance checks passed, but nonnegative
16x16 spatial weights cannot match the comparator's TRAIN recovery mean even
with oracle GT. [External result](../benchmarks/cc_v7_external/report.md),
[projector falsifier](../benchmarks/projector_probe_report.md).

New fundamental assumption to challenge: illuminant error is an adequate
surrogate for the product's skin-color error. It is not established, and the
user explicitly requests skin accuracy. First direct pilot now uses real
spectrophotometer-paired facial regional RGB/XYZ. Its source-CV-selected higher
order mappings lose to affine mappings on the held-out skin sites. Simpler
calibration remains a necessary baseline for any proposed neural mechanism.
Exact perceptual scoring requires a verified reference white, still absent.
The attempt to recover it from a companion spectral release failed; do not
substitute a convenient spectrum. [Skin target](skin_color_target_2026_09_11.md).

Retain signed-projector work as an alternative, but no major new angular-only
training cycle before resolving the direct skin measurement benchmark. The
older planned/in-progress labels below are historical.

User-directed broader search; keep current V6 combination as one hypothesis.
At each substantial result record an assumption to invert, a counterexample,
and a cheap discriminating experiment. No unusual name constitutes novelty.
Allocation for this phase: finish frozen phone audit; run V6 combination;
run a cheap non-CNN representation control; investigate prediction sets.

|Direction|Mechanism and inverted assumption|Potential advantage|Most likely failure|Cheapest falsifier|
|---|---|---|---|---|
|Canonical physical evidence (V6, running)|Remove absolute color coordinates from learned context and query evidence; only residual actions are learned|Gain-equivariant context and risk with same3.097M weights|Camera spectral changes are not diagonal; removing global color loses useful priors|Four matched source arms then3seeds; none-frame exact replay|
|Fourier correlation filter (next bounded spike)|Remove spatial CNN/backprop optimizer: color changes become translations of log-color histograms; solve ridge normal equations per Fourier frequency|About12k real filter/bias degrees of freedom; cheap global optimum for quadratic training objective|Histograms discard semantics; torus aliases; rare scenes violate learned reflectance prior|Train1126/validate119, fixed ridge sweep, compare preserved learned statistics/CNN controls; no phone tuning|
|Decision set instead of one illuminant|Keep multiple plausible corrections; accept only when all plausible corrections imply small color spread|Could reject ambiguity that point-confidence misses|Wrong prior excludes truth; source calibration does not transfer under shift|On source calibration, measure GT-set coverage versus corrected neutral-color diameter; only then freeze unseen-camera test|
|Distributed local constraints|Replace one global illuminant with a graph of locally compatible color hypotheses and robust consensus|Mixed-light detection rather than a forced global correction|Material changes masquerade as light changes; global-chart GT cannot validate a spatial field|Synthetic mixtures only test algebra; real single-chart benchmark can test refusal/diagnostics, not local color accuracy|
|Eliminate illuminant output entirely|Optimize correction decisions against calibrated downstream risk rather than recover latent light|Avoid spending capacity on nonidentifiable nuisance variables|Public illuminant GT supplies only a neutral-surface proxy; arbitrary Lab truth absent|Compare action-risk training versus point objective with identical capacity; existing V5 already shows negative transfer|
|Semantic teacher only at training|Transfer reflectance/scene priors to small student, then remove teacher|Potentially disambiguates dominant scene colors with same inference budget|Teacher color invariance erases photometric evidence; pretraining overlap; known KD prior art|Teacher/no-teacher feature target ablation on same real train/val; no phone feature extraction before lock|

Fundamental counterexample: in diagonal image formation I=R*E, for any positive
D, R'=R/D and E'=D*E produce exactly the same I when admissible reflectances
remain in range. A single image cannot determine universal absolute surface
color without priors/reference constraints. This does NOT require stopping R&D:
it motivates explicit ambiguity and support limits. A point estimate or more
iterations cannot remove this identifiability ambiguity.

Adversarial read of present evidence: V2 pooled phone advantage is not Samsung
uniform and its paired confidence interval includes zero. V5 corrected-color
critic has worse phone selective risk despite a source improvement. Therefore
physical-looking computation is not automatically better calibrated. V6 might
inherit both benefits, or inherit source bias and discard useful information.
Keep data/GT support separate from representational claims.

Prior art consulted freshly: Barron2015 Convolutional Color Constancy
https://openaccess.thecvf.com/content_iccv_2015/html/Barron_Convolutional_Color_Constancy_ICCV_2015_paper.html;
Barron/Tsai2017 FFCC https://research.google/pubs/fast-fourier-color-constancy/;
Multi-Hypothesis2020 https://arxiv.org/abs/2002.12896;
Uncertainty Estimation2025 https://doi.org/10.1016/j.patcog.2024.111175.
Histogram localization, posterior uncertainty and cascades already exist.
No conformal guarantee is claimed under nonexchangeable unseen-camera shift.

## Consequences of the next completed screens

V6 all three seeds are now negative against V5 on reused source validation.
The first V7 five-arm screen finds no benefit from canonical versus raw semantic
teacher targets: full2.4591 versus2.4349°, selective80 1.8630 versus1.7608°.
Sensor augmentation itself strongly improves fixed virtual mixing robustness,
but a preserved SoG model is stronger on the most severe mixing case. Keep all
five arms through the planned three-seed/real-transfer comparison; do not call
an artificial camera-matrix stress test a real unseen-camera result.

An assumption worth removing next is the need to choose a particular three-color
frame (which failed in V3). For a full-rank N-by-3 patch-color matrix X, the
projector P=X(X^T X)^-1 X^T is unchanged by X->XM for invertible3-by-3 M.
Project fixed spatial probe vectors through P rather than choosing a basis;
a compact network could predict invariant spatial weights and map them back
by X^T w. Compute P times probes through small solves, not a dense N-by-N P.
This removes arbitrary frame selection, not the physical identifiability limit.

Key assumption: a stable full-rank linear camera relation and informative
camera-invariant spatial structure exist. Advantage: exact algebraic response
to full linear mixing without camera ID or a selected color triple. Failures:
rank-deficient scenes, lighting/shading changes outside the global transform,
lost chromatic priors, and an illuminant outside the cone of observed colors if
weights are nonnegative. Cheapest falsifier before another neural architecture:
on TRAIN only, measure rank conditioning and a nonnegative patch-cone oracle
bound. If that bound cannot reach current accuracy, a positive-weight design
is ruled out; signed weights need their own positivity/refusal analysis.
This spike is PLANNED, not implemented or measured; V7 remains in progress.

Prior art limits the interpretation: [Color Homography](https://arxiv.org/abs/1605.04250)
and [Color Homography: Theory and Applications](https://ueaeprints.uea.ac.uk/id/eprint/65088/)
already connect changes of illumination/device to projective color mappings.
[Self-Supervised Learning of Color Constancy](https://arxiv.org/abs/2404.08127)
studies illumination-invariant representation learning. A projector is standard
linear algebra, not a novelty claim. Full geometric/prior-art review remains
necessary before promoting any particular projector-based learned mechanism.
