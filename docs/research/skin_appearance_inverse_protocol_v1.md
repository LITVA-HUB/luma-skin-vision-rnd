# Frozen conditional appearance inversion screen

Actual original MSKCC skin photographs and instrument-native Lab only. Existing
TRAIN/source VALIDATION roles; exposed independent TEST/CAL and external reserved
sets remain unread. Original MSKCC CC-BY, no new data/weights, no synthetic labels.
Source validation has been heavily explored, so all findings are exploratory.

## Change the direction of learning

Learn appearance statistics x given a measured skin color y, then infer y by
marginalizing nuisance capture modes. The likelihood is statistical, not a
validated physical camera renderer. No camera/mode identity enters inference.

Two input representations: three mean RGB channels (existing token columns
9:12 averaged over 64 patches) and all 18 mean patch statistics. Both use the
same source photographs, with no new pixel decoding. Camera protocols also
change people and capture composition; they are not pure camera interventions.

Forward mappings: standardized ridge degree 1 or degree 2, with either one
global component or four observed capture-mode components. Complete quadratic
terms include squares and interactions. Bias is unpenalized. Alpha candidates
are fixed at 0.01, 1, 100. Input/output mean/std are fit on the relevant TRAIN
rows, with 1e-6 std floor. Every person is excluded in turn to predict their
appearance from their measured Lab; fold mapping/scales use other people only.
These predictions estimate forward residual second moments, not a fitted skin
error head. Residual covariance uses 0.8 full second moment + 0.2 diagonal +
1e-4 identity in full-TRAIN standardized appearance units. Residual means are
not subtracted: systematic forward bias is included in the noise approximation.
All folds/OOF predictions remain reproducible. No source-validation residuals
fit the model or its covariance.

The skin prior is uniform over distinct TRAIN sites' actual native Lab atoms
(one per site, not per photo). Capture components have uniform prior weight.
Evaluate the Gaussian appearance likelihood with determinant normalizers, sum
over capture modes, then normalize over skin atoms using log-sum-exp. Two
fixed outputs: posterior mean Lab and posterior expected-DeltaE00-minimizing
palette medoid. Genuine same-convention native Lab supports these DeltaE00
costs. This does not create new ground-truth observations or a dense skin map.
Finite support can limit gamut; a color-prior/nearest-palette diagnostic must
be separated from image-model performance. The medoid is not a continuous
global Bayes optimum. Posterior mean is in the TRAIN palette convex hull.

## Baselines and budget

Direct standardized ridge x->y of degree 1/2, with the same two input views and
three alphas. Include the constant TRAIN-site-mean color diagnostic/control.
For each protocol: 24 forward fits + 12 direct fits = 36; all three protocols
mixed/from_SLR/from_ipod = 108 closed-form fits. They have no random seed search.
Forward fitting additionally estimates nuisance and OOF noise; direct fitting
is a simpler compute/data-label control, not equal-cost neural C+. Compare also
the archived compact neural source results, explicitly noting their richer
64-patch inputs and different capacity/training budget.

For each family (input kind, mapping direction/grouping, degree, output rule),
select alpha by smallest same-camera source-validation patient-mean DeltaE00,
breaking ties in listed alpha order. Score unseen-camera inputs afterward. All
candidates are retained transparently; do not pick alpha on the unseen camera.
All 108 fits run regardless of early results. No stochastic-model ensemble.

## Accuracy, risk and identifiability

Record actual skin mean/median/p95/tails and full curves at 100/95/90/80/70/60%.
Use a common nearest-TRAIN-input distance ranking within each input kind, so
all methods compared at a given kind accept the same images. Also score forward
posterior expected DeltaE00 as a separate uncalibrated risk hypothesis. Neither
ranking is calibrated C+, nor does a sharp posterior imply an error guarantee.
Do not report unitless novelty distance as predicted skin error MAE.

Record posterior effective atom count and likelihood evidence; check diffuse
or unsupported predictions rather than treating forward reconstruction as
inverse accuracy. A nearest-true-target palette oracle is a representation
diagnostic only, never an inference algorithm or a selection input.

A single linear-Gaussian model with Gaussian prior yields ordinary Gaussian
conditional regression; reversing its notation cannot create a new method.
Empirical color atoms and nonlinear/multimodal likelihoods test a different
restricted model but remain established Bayesian/inverse-regression mechanisms.
Novelty and phone precision are unproved. Prior art includes Gaussian mixture
regression and Bayesian color constancy; no external implementations adopted.

## Verification and next decision

Freeze code/protocol/data bindings before fits. Refit closed-form models and
OOF predictions exactly; independently check normal equations, Gaussian density
and palette expected costs, scalar CIEDE2000, selection and coverage. Check
person exclusion and exact native references. Retain parameter/storage/runtime
counts, but do not claim GPU latency for this CPU falsifier. Optimize/export
only after positive evidence. A failed inverse does not prove every neural
renderer impossible; a good forward fit does not justify building one either.
