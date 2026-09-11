# Frozen conditional patch-distribution falsifier

Endpoint: actual MSKCC instrument-native skin Lab / DeltaE00. Existing real
TRAIN/source VALIDATION photographs only. Exposed TEST/CAL and external reserved
sets remain unread. Original MSKCC CC-BY; no new external models/data or cloud.

Use the RGB means of the 64 observed image patches, token columns9:12. Native
Lab conditions an appearance distribution; it is a site reference, not a
per-pixel measured Lab label. No camera or capture-mode identity is used in
training or inference. Latent patch components are fitted from appearance
residuals rather than assigned to known capture labels.

## Models and matched input controls

Train on either the actual patch bag or one image-mean RGB vector. Conditional
mean maps use complete degree1/2 polynomials of standardized site Lab. Compare
one Gaussian component with three latent regression components. One-component
fits are deterministic (seed0, one fit), and three-component fits use seeds
17/29/43. Two representations x two degrees x (one+three initializations) x
three protocols =48 fits. Do not call deterministic duplicates independent seeds.

Input/output scales fit only relevant TRAIN inputs. Initialize three components
with single-start KMeans on residuals from the global conditional regression.
Use35 responsibility/update iterations, alpha1 ridge with unpenalized bias,
image-normalized sufficient statistics (each image total mass1, irrespective
of its patch count), covariance0.9 full+0.1 diagonal+1e-3 identity in standardized
RGB, and add-one smoothed component priors. Record all training likelihoods;
regularization/shrinkage means monotonic exact-EM likelihood is not guaranteed.

This is maximum-likelihood appearance noise, not a subject-held-out calibrated
color-error head. Its posterior-derived expected error remains an uncalibrated
hypothesis. Do not claim noise calibration from these training residuals.

## Inference and temperature control

For every actual TRAIN-site Lab atom, evaluate Gaussian-mixture RGB likelihood
at each observed patch. Average patch log likelihood, then exponentiate with
fixed strengths1/4/16/64 and uniform prior over TRAIN sites. Strength64 is the
independent-patch product; lower strengths temper correlated evidence. Select
strength only by same-camera source-validation patient-mean DeltaE00 for each
fit, with ties in listed order. Other-camera evaluation happens afterward;
all strengths remain archived. Predict posterior mean native Lab, with expected
DeltaE00 computed against genuine same-convention measured TRAIN color atoms.
Finite-support/gamut limitations remain; no corrected per-pixel image is claimed.

Also collapse patches to their original image mean at inference under the same
fitted model and same selected strength. This is an information diagnostic,
not an independently trained baseline. The separately mean-trained models are
the matched input-representation controls. For a single fixed-covariance
Gaussian, posterior scores of full bag and collapsed mean must agree up to
numerical precision. The non-Gaussian model must demonstrate something beyond
that identity or a temperature change before further architectural investment.

## Evaluation and boundaries

All48fits and all192 strength endpoints run. Mixed/from_SLR/from_ipod use existing
person-disjoint source roles; device transfer also changes people. Compare all
three random initializations without choosing one on transfer errors. Source
validation is heavily reused, so no confirmatory result follows.

Full/100/95/90/80/70/60%mean,median,p95,tails and curves use common TRAIN-RGB
novelty within protocols; additionally score posterior expected-error ranking.
Neither is matched calibrated C+ or a per-image error bound. Compare archived
direct/inverse RGB controls and stronger compact full-patch image models.

Verify exact model/refit arrays, responsibilities/weighted normal equations,
independent Gaussian-mixture likelihoods, single-Gaussian sufficiency identity,
scalar color metrics, temperature selection and preserved patient separation.
The component is a small CPU statistical experiment, not a GPU latency result.
Patch mixtures, mixture regression, Bayesian inversion and tempered/composite
likelihood are established mechanisms. No novelty or phone-color claim yet.
