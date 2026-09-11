# Decision: partial support-allocation signal, no innovation victory

21 fixed-budget models are complete:15 sampling fits and six post-screen
person-mass controls. Original MSKCC images and actual instrument-native Lab;
18 TRAIN people and232 images from six internal holdout people. Only original
TRAIN cache was loaded. These people participated in TRAIN in older experiments,
so this is exploratory mechanism evidence, not a new independent accuracy result.
[Full metrics, curves and controls](../benchmarks/skin_color_sampling_v1/report.md).

## What improves, and against which control

Mean skin DeltaE00 changes6.1082(image-uniform) to6.0012(color-weighted sampling).
Mean per-fit p95 changes14.9960 to14.2733; at80% accepted coverage,5.3918 to5.3187.
But ordinary person/site balancing gives mean6.0084, p9514.3638 and80%5.3496.
The color-versus-person/site mean difference is only-0.0073, while the descriptive
patient-balanced interval[-0.0624,0.0743] crosses zero. The color-versus-image
patient interval also crosses zero. No convincing advantage over this strong
simple comparator has been established. All figures average three separate runs.

Site-uniform sampling alone is worse at6.2240. Color-biased sampling with exact
importance correction to the site-uniform objective gives6.1729. Same draws,
different objective: this weakens an explanation based only on oversampling
stochastic efficiency. It does not establish the mechanism for every dataset.

## Attack the color-specific interpretation

Color weighting can incidentally change how often each person is seen. Two
extra controls therefore preserve every person's exact total probability.
Equalizing that person's site weights scores6.1407; shuffling those weights
among their sites scores6.1522. Original color allocation beats both in all three
paired seeds. Against within-person shuffle, the descriptive patient interval
is[-0.2927,-0.0058]. This is a limited signal that the within-person association
with color matters here; it is post-screen, one reused six-person cohort and
not multiplicity-adjusted. It is not independent confirmation or patent evidence.

The null that only person probability explains everything is weakened by these
controls. The stronger objection survives: plain person/site balancing already
matches the overall accuracy. Both explanations must be tested in wider source
protocols before promoting a method. No performance on ordinary phone selfies
is inferred from these clinical skin images.

## Next bounded experiment and combination

Combine the two partial leads explicitly: uniform total mass per TRAIN person,
but distribute that person's mass over their sites in proportion to the frozen
color proposal. This preserves within-person color emphasis and avoids letting
the number of sites determine a person's overall training importance. No person
or camera identity is needed at inference; these are training sampling metadata.
This is an unverified hypothesis using established balancing mechanisms.

Before fitting, freeze a wider source screen: image, person/site, site, color,
color-IPW and the combined sampler; same original core, same930 updates, three
seeds. Use original TRAIN and reused source VALIDATION for mixed and both
camera-held-out training directions. Retain known-camera and unseen-camera
metrics separately for the single-camera models, fixed final checkpoints,
and all negative seeds. Do not use held-out-camera scores to choose weights,
bandwidths, checkpoints or thresholds. No exposed TEST/CAL access.

The mechanism to test is allocation of scarce gradient budget to independently
measured surface colors across people, not camera calibration. Failure modes:
rare-reference noise, common-color accuracy loss, unstable tail gains and stronger
capture/population confounding under transfer. The cheapest falsifier is this
matched source screen before any architecture expansion or deployment work.
Do not call balancing novel: continuous-label smoothing and balanced regression
are already established; see the current prior-art addendum.

## Evidence boundaries

21 exact prediction arrays; three historical control states exactly reproduced;
four extra full state refits;4872 scalar color cases;750141 scalar TRAIN density
pairs;126 fixed-coverage rows;108 person-mass identities. Core929,297 parameters;
maximum observed fit allocation288.35MiB. No new isolated inference benchmark.

Independent MSKCC stays primary4.4570/80%4.1591 vs ordinary fusion4.3005/4.1447.
No new strongest independent-baseline win, calibrated refusal guarantee, universal
camera model or facial-phone proof. Original MSKCC CC-BY; no new outside data or
weights, paid resources or publishing. Goal remains active and unmet.
