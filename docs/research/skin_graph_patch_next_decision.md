# Skin color endpoint: graph/support and patch-distribution decision

Actual instrument-referenced skin Lab and CIEDE2000 remain the primary endpoint.
Illuminant angular error cannot establish skin-color accuracy. This milestone
contains 36 neural graph/support fits and 48 statistical patch-likelihood fits
on existing original MSKCC source photographs. No new independent test was run.
The repeatedly reused source validation cohort makes these exploratory results.

## Combination result

The graph is applied only to the original spatial grid during training. The
second pass, including same-site mixed patches, always uses the plain core.
Unregistered captures are never presented as one registered spatial grid.
Inference is single-image with no camera identity and 924,932 active parameters.

| Source protocol | New plain control | Graph plus support | Stronger historical control |
|---|---:|---:|---:|
| Mixed | 3.4855 | 3.5143 | 3.4406 |
| SLR to iPod | 5.6063 | 5.2801 | 4.8328 |
| iPod to SLR | 6.1399 | 5.6610 | 4.9736 |

Values are mean skin DeltaE00 averaged across three seeds, not ensemble scores.
The combination helps the new controls on transfer but harms mixed accuracy.
Against paired-only controls, transfer differences are -0.2004 and -0.1721;
descriptive patient intervals are [-0.3579,-0.0473] and [-0.3209,-0.0135]. These
small reused cohorts and multiple explored comparisons do not confirm novelty.
Historical graph training used a different schedule; this is not its exact refit.
At common 80% coverage the combination scores 3.4239 / 5.1110 / 5.9994.
Reverse selection is worse than accepting every image. Input novelty is not a
calibrated expected-error head. Camera and participant composition are confounded.
[Full graph report](../benchmarks/skin_graph_support_v1/report.md).

## Changed representation result

The alternative learns patch appearance conditional on actual TRAIN site color,
then infers color over the TRAIN palette. It uses no camera or acquisition-mode
labels. Three latent Gaussian components replace one Gaussian; degree one/two
appearance functions and image-mean versus full-patch inputs are controlled.
Site reference Lab is not a measured per-pixel map.

Mixed degree-two full-patch mean error improves from 4.7854 with one component
to 4.0554 with three. But three-component image-mean training scores 4.0120,
and the historical image baseline remains stronger at 3.4406. The corresponding
full-patch transfer scores are 6.6379 and 6.6877. This is not a universal gain.
All 48 fits select likelihood strength 1 from the frozen list 1/4/16/64 using
same-camera validation. Processing 64 patches as independent evidence is not
supported by this screen; smaller strengths remain untested, not proven optimal.

All 12 one-Gaussian fits obey the sufficient-mean identity to 2.84e-14. With
three components, full-patch versus collapsed inference can differ, but the
mixed degree-two difference is only about -0.0030 DeltaE00. Changing confidence
or processing more correlated observations must not be called new color signal.
[Full distribution report](../benchmarks/skin_patch_likelihood_v1/report.md).

## Next bounded decision

Retain the graph/support partial gain as a control, not the chosen invention.
Before another output-head sweep, test whether population support or the input
representation limits skin accuracy. Use nested TRAIN-person subsets, fixed
validation roles, equal optimizer-update budgets and multiple subset seeds;
compare the strongest compact patch baseline with a capacity-matched learned
pixel representation. Keep all exposed TEST/CAL data out. The cheap falsifier is
a learning-curve screen on TRAIN-person folds before full fitting. Record patient
counts, site counts and color-support changes separately from camera labels.

Alternative mechanism: learn relationships among local skin appearance and
context rather than an acquisition identity. The assumption is that the image
contains enough reference-independent information to separate surface color
from capture effects. The benefit would be one-image inference without camera
metadata; the likely failure is non-identifiability or context shortcuts. The
cheapest intervention is to remove/shuffle context under the same frozen model
and compare actual color errors, then require retrained matched controls before
claiming context causes a gain. This is an unverified hypothesis, not an invention
or a promise that any camera can be normalized from any JPEG.

Victory requires lower actual skin DeltaE00 and better calibrated color-error
selection than the strongest matched baseline on genuinely new people and unseen
ordinary phone acquisition. The working product aspiration is median <=2 and
p95 <=5 at >=80% accepted coverage, subject to application validation; it is not
a universal perceptual threshold. That aspiration remains unmet.

The archived independent MSKCC result stays primary4.4570/80%4.1591 versus
ordinary fusion4.3005/4.1447. Clinical/dermoscopic skin photographs do not validate
ordinary facial selfies, cosmetics matching or unrestricted camera generality.
No new data, outside weights, paid resources, export or external publication.
