# Mechanism search ledger: challenge the representation

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
