# Decision after 36 compact skin-color density fits

Do not promote the four-component density or claim a universal color-accuracy
gain. All results are real-image instrument-Lab SOURCE development scores on
repeatedly explored cohorts. The independent test remains unchanged and exposed.

## What the comparison actually found

Mean DeltaE00, averaging three separate seed scores:

| Method | Mixed | SLR to iPod | iPod to SLR |
|---|---:|---:|---:|
| Ordinary strong control | 3.4406 | 5.0193 | 5.9635 |
| Ordinary without mode auxiliary loss | 3.4694 | 4.9838 | 6.3885 |
| Single Gaussian mean | 3.4605 | 5.6571 | 5.7474 |
| Four-component density mean | 3.5484 | 5.4148 | 6.4005 |
| Density minimum-risk finite decision, order3 | 3.7181 | 5.4146 | 6.8962 |

The prior training-only graph still has stronger reverse mean4.9736. Removing
mode supervision alone is not a universal fix. The density's mean and finite
minimum-risk decision do not establish a gain. This does not refute every
conditional distribution family or a continuous exact Bayes decision.

Useful partial evidence: mixed Gaussian mean error at80% predicted-risk coverage
is3.2709 versus ordinary control3.4407 with dispersion ranking. Across all images
Gaussian is slightly worse (3.4605 versus3.4406). Its risk benefit is not universal:
SLR-to-iPod80%5.4955 versus5.1087; reverse5.9326 versus5.7914. Confidence can fail
under domain shift. This is not calibrated C+ or a per-image guarantee.

[All fits, curves and descriptive intervals](../benchmarks/skin_distribution_v1/report.md).

## Numerical opponent check

Order2 versus order3 integration changes density decisions. A separately frozen
post-hoc antithetic Sobol1024/4096 diagnostic tests the same candidate set on all
18density fits. It cannot change weights, the primary endpoint or the candidate
family. Preserve these sensitivity results separately; approximation failure
must not be confused with a proof against distributional modeling.

Completed diagnostic:4096-node mean errors for the mixture decision are
3.5993/5.3897/6.8491 across mixed/SLR/reverse, still no strongest-control win.
Corresponding1024-node means are3.6015/5.3926/6.8491. Additional6,876scalar cases
pass. Thus order3approximation contributes to the measured error, but improving
integration alone does not rescue this fitted density and candidate family.
No general impossibility or exact-integral claim follows.

## Next competing experiments

1. Cross the ordinary predictor with Gaussian-derived color-risk information.
   Compare expected DeltaE00 around the ordinary prediction, covariance-only
   uncertainty, inter-model distance, and ordinary dispersion. Include a
   capacity-matched two-model ordinary ensemble and C+ error head with strictly
   person-held-out residuals before claiming a selective mechanism gain.
   Hypothesis: the strongest color mean and most useful uncertainty may come
   from different objectives. Likely failure: confident shared camera bias.
   Cheapest falsifier: frozen source-only risk crossing, without retraining.

2. Remove density estimation itself. Learn conditional expected DeltaE00 as a
   function of image and candidate Lab, then choose a low-risk candidate. The
   risk target can be computed from actual TRAIN instrument Lab without inventing
   references. This avoids estimating an entire density that is only used through
   one loss integral. It is related to established conditional risk minimization,
   not inherently novel. Likely failures: risk-surface smoothing, discretization
   bias, and optimistically exploiting extrapolation. Compare matched ordinary
   regression and density models and verify candidate-gamut error separately.

Neither branch substitutes confidence for accuracy. Success still requires
better actual skin DeltaE00 and reliable accepted-image tails, followed by a new
untouched compatible benchmark. Ordinary phone facial validation is outstanding.

## Evidence and deployment scope

935,453stored parameters (929,297active in MSE arms). All36fits share shapes and
initial states. No camera identity or foundation model is needed at inference.
Numerical integration adds CPU work; no batch1latency/export claim is made.
108exact color replay arrays,12,672independent scalar color cases,432coverage
rows and51,660manual integration scalar cases pass. Full suite322tests passed
with14historical warnings in34.33s. Tests verify implementation, not superiority.

Original MSKCC CC-BY only; this branch imports no CIE/ISSA buffers or third-party
weights. The earlier material branch's separate CC BY-SA obligations remain.
No files or claims are published externally. Goal active and not achieved.
