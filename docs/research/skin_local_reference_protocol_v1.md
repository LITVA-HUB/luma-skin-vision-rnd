# Frozen person-excluded local reference regression

Real original MSKCC CC-BY photographs and native instrument Lab. Original TRAIN
24people/966images only; every evaluated person excluded from reference bank,
target/input scales and all regressions. Original TRAIN has been heavily reused:
this is exploratory source cross-validation, not new independent confirmation.
No source VALIDATION, CAL, TEST or outside held-out dataset is read.

## Hypothesis and counterexamples

Comparative recognition alone failed to establish absolute color accuracy.
Here query-dependent support weights may adapt a local appearance-to-Lab
mapping. This is standard locally weighted learning, not claimed novelty.
Failure: appearance proximity tracks capture or texture, predicted-color
proximity reinforces an already wrong color, sparse support amplifies noise,
or reference averaging shrinks uncommon colors toward common ones.

## Fixed input, roles and six methods

Only existing color36 image descriptors enter inference. For each of24 sorted
TRAIN people, use all images of other23 people as bank. Original per-photo
weighting is preserved; repeated sites/people are not independent measurements.
Use float64. Standardize descriptors by bank mean/std, floor1e-6. Center native
Lab by bank mean. Global ridge solves (X'X+I)B=X'(Y-center); intercept is not
penalized. Its predictions must replay prior skin_relational_probe ridge36.
The global mean must replay the prior excluded-person mean control.

1. global_ridge: the exact global linear control.
2. global_mean: the exact other-people mean color control.
3-4. appearance_mean/appearance_affine: Gaussian log weight -0.5*d^2 where d
     is RMS distance in36 standardized descriptors, fixed bandwidth1.
5-6. color_mean/color_affine: Gaussian log weight -0.5*(d/5)^2, where d is
     genuine CIEDE2000 distance between global-ridge estimated query/bank Lab.
     Fixed bandwidth5; query reference color is never available to the function.

Weights are computed with max-subtracted exponentials and normalized to sum N,
the number of bank images. This fixes ridge penalty scale across affinities.
No tuned bandwidth, chosen subset, target-camera calibration, learned encoder,
clipping or per-person/camera identifiers at inference. Camera labels are not
used to construct an affinity. The color kernel uses estimated colors for both
query and bank, never a mixture of query estimate and bank instrument labels.
Bank references are used as regression/mean targets only.

Weighted mean predicts sum(w*y)/N. Local affine computes weighted input/target
means, then solves centered (X'WX+I)B=X'WY and predicts ymean+(query-xmean)B.
It may extrapolate. The uniform-weight limit must recover global ridge for
every query. No renamed ensemble or neural architecture is asserted.

There are24 global ridge fits,1932 local-affine solves and1932 weighted means,
plus exact global controls;6 systems with966 person-excluded outputs each.
All configurations are retained regardless of outcomes; no post-hoc selecting
queries or tuning parameters on these results.

## Actual skin endpoint and support diagnostics

Score genuine native instrument Lab with CIEDE2000: mean,median,p95, fraction>10,
equal-person and equal-site means. Compare each method with global ridge on
the same images; count improved people. Do not describe folds as independent
training repetitions or claim significance from966 correlated photographs.

Use one shared, fixed novelty score for all six methods: nearest bank RMS
descriptor distance after bank-only normalization. Report full risk-coverage
curves and100/95/90/80/70/60% exact coverage. The accepted images must be identical
across methods. This isolates color mapping; it is not a calibrated confidence
head, a matched learned C+ or a deployable per-image error bound. No selection
threshold is fitted here. Record effective weighted photo support N^2/sum(w^2),
not a count of independent people, and mass allocated to bank people.

## Reproduction and decision

Freeze code/tests/protocol, cache and previous ridge/mean output hashes before
running. Keep banks, weights, predictions and identifiers private under ignored
experiments. Independently replay Gaussian weights, augmented weighted least
squares, exact role exclusion, target perturbations and scalar CIEDE2000.
Verify original caches/checkpoints are unchanged; no new neural training here.

Assess actual mean and tail color error, not support matching alone. A source
gain justifies an explicit known/unseen-camera follow-up with strong compact
comparators and matched C+, not novelty or a new independent-phone claim. If
local models lose global ridge, record the failure before increasing complexity.
Any eventual inference bank must count all descriptors/references/scales and
base coefficients in file size and runtime. No new RTX/ONNX/TensorRT claim is
made by CPU linear algebra or cached feature timing.
