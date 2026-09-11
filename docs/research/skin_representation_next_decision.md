# Skin color remains the endpoint: representation controls

The product endpoint is agreement with an actual skin color reference in native
Lab, scored with CIEDE2000. Illuminant angular error cannot substitute for it.
The preserved independent MSKCC result is mean 4.4570 DeltaE00, or 4.1591 at
80% exact coverage. Ordinary fusion remains stronger at 4.3005 / 4.1447.
The proposed risk-head advantage over C+ is not statistically convincing.
These are clinical capture images, not an ordinary facial-phone validation.

## Completed source experiments

Two predeclared families contain 72 fits in total: four arms, three seeds,
mixed-camera fitting and both source camera-transfer directions per family.
All endpoints are original TRAIN/VALIDATION only. Camera families and validation
results have repeatedly been examined; these are adaptive source screens, not
a fresh independent test. Means below average seed scores, not predictions.

| Method | Mixed mean DeltaE00 | SLR to iPod | iPod to SLR |
|---|---:|---:|---:|
| Historical capture plain | 3.4771 | 5.8301 | 5.5089 |
| Historical capture mixture | 3.4406 | 5.0193 | 5.9635 |
| Training-only learned graph | 3.6394 | 5.8339 | 4.9736 |
| Training-only fixed grid | 3.6128 | 5.2823 | 5.1586 |
| Training-only complete graph | 3.7441 | 5.2972 | 5.4993 |
| Training-only constant residual | 5.5085 | 6.4028 | 8.5210 |
| Distribution model, no histogram information | 3.4801 | 5.7770 | 5.9897 |
| Distribution model, ordinary RGB histogram | 3.6507 | 5.6347 | 7.9333 |
| Distribution model, copula context + absolute color | 3.6460 | 6.0339 | 5.5702 |
| Distribution model, ranks only | 5.0899 | 8.8349 | 6.1987 |

### Assumption attacked: learned relations are necessary

Fixed grid and complete graph controls explain some transfer gains without
learned affinities. Learned graph remains better in the reverse direction,
but beats fixed grid in only two of three seeds there. Constant residual does
not explain the effect. All deploy the identical 924,932-parameter inference
core. No universal advantage or graph-specific novelty is established.

### Assumption attacked: removing camera-sensitive coordinates helps color

Rank dependence is exactly preserved under strictly increasing per-channel
transforms in the declared probes. Independent feature checks reconstruct all
1,230 source histograms and 1,536 patch profiles exactly. Channel mixing and
clipping violate that limited invariance. The matched RGB-histogram and copula
models have 1,072,644 parameters; no-histogram nominal capacity includes a
constant-input branch and is not equally informative capacity.

Rank-only color prediction is worse than the absolute-color control in every
protocol. Adding copula context helps one transfer direction relative to its
own control but loses to the strongest historical comparator. This falsifies
this particular implementation as a universal improvement. It does not prove
that every invariant representation must fail. Reject this candidate for
promotion; preserve absolute skin lightness and chromatic information.

## Selective accuracy and integrity

Copula mean at 80% coverage is 3.7053 / 6.2065 / 5.5179. For the first two
protocols rejection worsens error. Patch dispersion is uncalibrated and is not
a reliable expected-error estimate or a production acceptance threshold.
Full curves and all six fixed coverages remain in both reports.

Across both families, 216 color arrays and 216 risk arrays exactly replay;
12,672 independent scalar CIEDE2000 cases and 432 coverage rows pass. Nine
learned-graph controls reproduce the previous final weights exactly. The
nuisance family includes a documented test-only precision amendment: a
floating-point invariance test changed to fixed-seed FP64 after a float32
tolerance failure. Original test and lock are archived. Model, data, optimizer
and selection bytes were unchanged; do not conceal the amendment.

## Next research decision

Neither family earns a new accuracy or novelty claim. Do not export or scale
them simply because they are unusual. Keep the one-direction training-only
graph as a candidate and keep the stronger ordinary comparisons.

Before another architecture sweep, investigate whether image/reference
correspondence and capture-dependent ambiguity limit native Lab prediction.
The next bounded check should use only source measurements to quantify repeat
instrument-reading disagreement and paired-view prediction disagreement,
stratified by capture mode and reference lightness. Repeat-reading spread is
a diagnostic, not an asserted irreducible error floor. No new labels are to be
fabricated from illuminant vectors, and no exposed test is to be reused.

Mechanism hypothesis: shared color evidence across genuine paired captures
may separate measurement instability from image-dependent error. The critical
assumption is that site-level references correspond adequately to the visible
skin region. Failure mode: capture geometry or local heterogeneity violates
that correspondence. The cheapest falsifier is a source-only correspondence
and residual audit before fitting another model. This diagnostic is planned,
not completed or evidence of improved accuracy.

Ultimate success remains a reproducible reduction in actual skin DeltaE00
against strong matched controls, including p95 and fixed coverage on genuinely
held-out people and devices. Median <=2 and p95 <=5 at >=80% coverage is a
working product aspiration, not an achieved result or universal standard.
Ordinary iPhone/Android facial validation with suitable references remains
unmeasured; the current dataset cannot establish it.

[Nuisance controls](../benchmarks/skin_nuisance_v1/report.md),
[copula controls](../benchmarks/skin_copula_v1/report.md),
[preserved independent skin test](../benchmarks/skin_mskcc_selective_v1/report.md).
