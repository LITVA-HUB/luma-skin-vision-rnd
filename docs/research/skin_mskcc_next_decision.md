# Direct skin endpoint: next experiment decision

2026-09-11. The user's primary endpoint is skin-color accuracy. A lower
illuminant angle is neither the endpoint nor sufficient evidence for it.

## Measured starting point

MSKCC native instrument Lab is now available from the original CC-BY release.
The first locked candidate set used author image-color summaries. On six
validation people: MLP(64,32), alpha1 mean DeltaE00 4.3702, median3.8038,
p959.7150; polynomial ridge alpha1 mean4.4994. This is model-selection data,
not the still-sealed ten-person final test. The observed mean improvement over
polynomial regression is small; it does not establish a special mechanism.
Nearest-neighbor distance rejection barely changes mean at80% (4.3694) and
worsens p95 (10.4254). This is a negative reliability result, retained.

## Assumption to remove

Previously the pipeline treated a global illuminant estimate as the necessary
intermediate to trustworthy surface color. Remove that requirement. Predict
instrument-native skin Lab and its residual error directly, then compare
against an explicit-normalization branch only if it helps the actual endpoint.
Do not discard absolute brightness/color by imposing unconstrained invariance:
those channels contain the target information as well as camera nuisance.

## Smallest informative next experiments (PLANNED, not measured)

1. Recompute deterministic image features locally from the acquired full JPEGs;
   compare median-only, color-distribution and spatial/texture features using
   the fixed subject roles. This falsifies whether author preprocessing is
   necessary, and whether a single global color summary discards useful evidence.
2. Train a standard compact image baseline from scratch, within1–5M parameters,
   against native Lab. Use no unverified dermatology weights. Preserve all source
   outcomes and freeze final candidates before exposing held-out participants.
3. Compare a matched network with a training-only same-site consistency term:
   different acquisition modes of the same skin site should yield the same
   instrument color. At inference use one image. Hypothesis: physically paired
   views constrain nuisance variation more accurately than arbitrary color jitter.
   Main assumption: the repeated imaging modes do not change the relevant tissue
   state enough to violate label correspondence. Main failure: contact pressure,
   alcohol and polarization change observed tissue and suppress relevant signals;
   consistency can erase color differences or learn patient/camera shortcuts.
   Cheapest falsifier: same-backbone/same-budget paired versus unpaired training,
   validation by held-out person and acquisition mode; neither camera identity
   nor site identity may enter the inference network. This is prior-art-adjacent
   multi-view invariance, not a verified novel architecture.
   Mathematical adversary: with squared output error, the sum over same-site
   views decomposes into error of their mean plus within-site prediction
   variance. Adding squared output-consistency just reweights an existing
   term. Treat this arm as an ordinary control, not a fundamentally new method.
   A stronger mechanism must demonstrate additional information or useful
   constraints beyond that decomposition in a matched experiment.
4. Retain a radical comparator: color-distribution/set inference with no spatial
   convolution, versus texture-only inference with global image color removed.
   Their failure/success distinguishes reflectance evidence from camera/subject
   shortcuts. Texture-only cannot be called color measurement without direct
   target success and subgroup checks. Do not combine arms merely because each
   improved a different reused validation score; freeze and test the combination.

## Reliability endpoint

Predict expected DeltaE00 on source-subject-held-out residuals; calibrate only
on reserved calibration people. Compare with standard error head, density
distance, ensemble disagreement and no rejection at identical coverage. Same-site
triplicates quantify reference variability; do not subtract their errors from
model errors or present repeatability as a certified instrument accuracy floor.

## What would count as a win

An independent patient-held-out improvement over the strongest locally
reproduced compact baseline, especially at80% accepted coverage and in the
catastrophic tail; confidence intervals grouped by participant. Then a source-
camera-only trained model evaluated on the other camera's held-out people.
Clinical close-ups must be reported separately from dermoscopy. No universal
camera or ordinary selfie claim follows from two calibrated acquisition systems.
The aspirational medianDeltaE00<=2 / p95<=5 at>=80% accepted on new faces and
phones remains a proposed product target, not an achieved or established
perceptual/clinical standard. Reference repeatability makes those numbers
particularly important to validate rather than promise.

Prior evidence remains intact: synthetic neural model lost A2; V7 canonical
teacher lost matched C+ and Fourier; He XYZ pilot selected nonlinear controls
that lost affine on its original test. There is no project-wide novelty victory.
