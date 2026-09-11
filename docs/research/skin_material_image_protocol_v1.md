# Image-to-instrument color with a measured-material decoder, v1

Frozen before model fitting. Endpoint remains actual MSKCC native Lab and
DeltaE00, D65/10-degree. No angular-only optimization. Existing independent
TEST/CAL remain untouched; source validation is heavily explored and any new
gain is exploratory. No ordinary phone-selfie accuracy claim.

## Material source and color interface

Use only the already frozen ISSA TRAIN8680records and logit-reflectance PCA
basis, first8components. Calculate coefficient scales with subject weighting
on those TRAIN records only. Never use ISSA native2-degree Lab as an MSKCC
target. A latent8-vector decodes to31reflectances via sigmoid(mu+Bz).

Build D65/10-degree XYZ integration from original CIE1nm tables,360-830nm.
Linearly interpolate31samples400-700nm and extend endpoint reflectances
constantly outside400-700nm. This is an explicit model assumption, not
measured tails. Document its sensitivity on existing ISSA TRAIN records that
have extra measured bands. No simulated reference becomes a photo target.
NaN zbar10 entries above559nm are outside the tabulated column support; use
the official metadata's zero extrapolation there, not for missing reflectance.
Use full-table D65 white for the decoder's Lab. Geometry and spectral tails
remain approximate; the residual variants allow departures from this prior.

CIE original metadata licenses tables CC BY-SA4.0. Attribute CIE/DOIs and
preserve ShareAlike for adapted constants. ISSA is original CC BY4.0. This
does not declare an entire exported model proprietary-license cleared.
Original observer CSV matches metadata MD5/SHA256; the landing-page MD5 is
different and is recorded as a source metadata discrepancy, not hidden.

## Matched architecture and controls

Shared CaptureColor local18->256->256 and context768->512->512 backbone,
four hypotheses and learned patch/gate weights. Final patch vote layer has
45outputs:4x8latent,4x3free color coordinates,1patch weight. Identical stored
parameter shapes and initialization for all five arms. Inactive paths are
reported, not counted as required inference capacity.

- direct: four standardized Lab predictions from the free3coordinates.
- tangent: four8latent predictions decoded by the fixed first-order Taylor
  expansion of the material map atz=0, using its exact base color/Jacobian.
- material: full nonlinear sigmoid-spectrum-XYZ-Lab decoder.
- tangent_residual: tangent plus free standardized Lab residual.
- material_residual: nonlinear material plus the same free residual.

Material/tangent receive the same external material prior; tangent is a
strong mechanism control for local color geometry and initialization.
Compare residual versus residual, and hard versus hard. This does not yet
establish a skin-specific advantage over a random material basis. If positive,
that ablation and matched calibrated expected-error heads are required.

Loss: standardized native Lab MSE +0.1capture-mode cross entropy for all arms.
Residual arms additionally pay0.01mean squared standardized residual. Mode is
only an auxiliary training label, never an inference input. No camera ID,
calibration target, teacher model, adaptation or cloud call at inference.
Predicted four-hypothesis RMSLab dispersion is an uncalibrated diagnostic,
not an error bound. Report fixed coverages100/95/90/80/70/60and full curve.

## Training and evaluation

45fits:5arms x3seeds17/29/43 x3protocols mixed,SLR-to-iPod,ipod-to-SLR.
Same existing source image128cache and64x18patch summaries; same site-paired
sampler16pairs/32images,80epochsAdamWlr.001wd.01cosine to.00001.
Target mean/std from each fitting cohort only. Epoch selected by same-camera
source-validation patient-balanced meanDeltaE00. Cross-camera endpoints are
evaluated after selection only. No transfer-camera epoch/hyperparameter tuning.

Compare against all strong historical compact controls, separately labelled
locally reproduced earlier. Average three separate seed scores, not an
ensemble. Preserve every failure. No export/latency optimization before a
positive matched-control result. Report parameter count, file size, peak fit
GPU allocation; these alone do not prove batch1deployment latency.

## Hypothesis and falsification

A measured skin-material decoder may improve cross-camera estimation through
output geometry while remaining tiny. Assumption: the approximate material
family helps the image inverse problem more than a matched ordinary mapping.
Failure: source skin/material/geometry bias removes valid MSKCC variation or
the model learns only the residual. Cheap falsification: hard-vs-tangent and
residual-vs-residual comparisons above. If no consistent gain, reject this
decoder rather than claiming that low oracle spectral error was success.
Spectral PCA, inverse rendering and physical skin models are prior art. No
novelty claim is made merely from composing them with a compact network.
