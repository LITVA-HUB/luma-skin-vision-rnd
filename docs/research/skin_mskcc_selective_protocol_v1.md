# Selective instrument skin-color experiment v1

Frozen before out-of-fold risk fitting or opening CALIBRATION/TEST numerical
endpoints. Source architecture screens remain exploratory. Original subject
roles, image preprocessing and native instrument Lab references are unchanged.
Training24people/966images, validation6/264, calibration6/208, test10/400.

## Primary comparison, chosen before the test

Both C+ and Proposed use EXACTLY the same confidence-weighted patch color
estimator. Primary: arithmetic mean of seeds17/29/43 (2774796color parameters).
Secondary: seed17single model (924932parameters); seeds29/43 are replication
comparators. No new color training on validation/calibration/test people.
All source-selected best checkpoints are retained and hash-locked.

C+ predicts expected actual DeltaE00 from36global image-color features,3predicted
Lab channels and5-neighbor source density distance (40inputs). Proposed adds
8within-image witness statistics and1cross-seed disagreement (49inputs).
For single-seed models the cross-seed channel is zero. The color estimation
backbone, images, fitting labels, OOF folds, candidate heads and calibration
are otherwise identical. A standard MLP head has4737versus5313parameters;
the576extra parameters are explicitly recorded, not hidden as exact equality.
Additional disagreement features are an UNVERIFIED selective-color mechanism,
not a novelty claim about ensembling or uncertainty in general.

Witness features, in this fixed order: confidence-weighted RMS Lab residual;
p90and maximum unweighted Lab residual; normalized confidence entropy; maximum
confidence; norm of unweighted-minus-weighted mean vote; norm of bright-half
minus dark-half mean vote; effective sample fraction1/(64*sum(weight^2)).
Internal norms are Euclidean native Lab (DeltaE76 scale), NOT invented targets.
Ground-truth supervision is ALWAYS true CIEDE2000 to the measured site label.
Ensembles average the8witness features and append RMS dispersion of seed Lab
predictions. Bright/dark halves use mean RGB of the patch-median channels.

## Out-of-fold labels

Sort the24TRAIN patient IDs by SHA256("LumaMSKCCRISKv1|"+ID), split into6folds
of4people. For each fold and each seed17/29/43, train plain patch-vote model on
the other20people only, using the original80epoch recipe. Select epoch using
the fixed original6VALIDATION people. Fold target standardization uses20people
only. Only after checkpoint selection predict the4OOF people. No OOF target
is used for its color model, normalization or epoch selection. Every TRAIN
person appears in exactly one OOF target fold. Heads learn those OOF residuals;
do not use residuals of a color estimator fitted on the same person's labels.
Density features use only each fold's20fitting people; deployment density uses
the original24TRAIN people. This training-size shift is a limitation to report.

## Fixed matched risk-head search

For each color version (3single seeds and their ensemble), each feature arm
(C+/Proposed), fit the same three candidates on OOF data: standardized-input/
target MLP(64,32),tanh,L-BFGS,alpha1or10,max_iter2000,seed17; and HGB squared
error,100iterations,lr0.05,max_depth2,min_samples_leaf32,l2_regularization10,
seed17. Target is DeltaE00 directly, not log error. Clamp raw predictions to>=0.
Select each arm's head by VALIDATION actual mean error at80% coverage, breaking
ties by mean empirical risk over60–100% coverage, then candidate name. Keep all
candidate results. Selected models remain independent of calibration/test.

## Calibration and sealed test

Write a pre-calibration lock binding this protocol, source manifest/cache,
all inference/risk/evaluation scripts, color weights and risk-head weights.
Only then decode the6CALIBRATION people. Fit one nondecreasing isotonic mapping
from selected raw predicted error to observed DeltaE00 with out_of_bounds=clip,
weighting each calibration person equally. This is empirical calibration,
not a distribution-free guarantee. Fit thresholds for predicted error<=2/5
and raw-score quantiles at100/95/90/80/70/60% calibration coverage. No retuning
after seeing calibration results. Archive errors/rankings/calibration maps.

Then write a FINAL lock including the calibration objects/thresholds and all
pre-calibration bindings. Test loader requires this exact lock hash and verifies
all files before decoding any test pixels or numerical references. No selection,
refitting, exclusions or threshold changes after test. Fail on any nonfinite
prediction; do not silently discard bad cases. Return actual counts/support.

Report mean/median/p90/p95DeltaE00, fractions>5and>10, patient-balanced mean,
clinical versus dermoscopic and camera strata. Full risk-coverage curves and
fixed100/95/90/80/70/60% ranking metrics use ceil(coverage*N) and ties by raw score
then image ID. Isotonic plateaus must not arbitrarily destroy raw-score ordering.
Additionally report realized test coverage/risk at CALIBRATION-fixed thresholds
and at calibrated predicted error<=2/5. These are distinct from exact-coverage
test ranking curves. Report predicted-error MAE and fixed calibration-bin
agreement. Patient-cluster bootstrap2000,seed20260911 for paired mean and80%
risk differences; interval selection/multiple comparison limitations explicit.

Include all9local pixel controls and21neural color fits,7three-seed ensembles
and4fixed CNN/patch fusion comparators as color baselines; no author-derived
feature models masquerading as local image inference. Published paper numbers
are not locally reproduced. Primary C+ and Proposed differ in risk, so their
100%coverage color errors MUST be identical. Do not claim an overall color
accuracy gain from changing rejection alone.

No true new-camera result is possible from calling these two development
devices unseen. Report KNOWN-camera test people only. Ordinary phone facial
accuracy, clinical utility, instrument trueness and cosmetic shade matching
remain unvalidated. All source negatives remain preserved.
