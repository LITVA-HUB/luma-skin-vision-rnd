# He 2021 paired facial RGB-to-XYZ baseline, v1

Freeze this protocol and numerical scripts before inspecting Testing XYZ/RGB.
The original workbook has been inspected for text/header structure and subject
IDs only. Numeric training-chart data were used for a separate failed companion
spectrum correspondence check. No skin test errors have been measured.

Scope: real skin-reference calibration bootstrap, not Proposed, a whole-image
model, camera-independent validation, or a reproduced author accuracy number.
Original workbook SHA256:
e3ad5b30a828c542ba2d23dd7fe57cddf0c3de2a163f1815cd2e482a816484a8.
Original license CC BY 4.0; cite Ruili He, DOI 10.5281/zenodo.5532176.

Author split: FSCD rows3:202, subjects Obs.1 through Obs.40; Testing rows3:102,
Obs.41 through Obs.60. Five sites per person: FH, CBR, CBL, NT, CH. No subject,
location, ethnicity or gender is an input feature. RAW and JPG use separate
models. No full face images or guessed surface measurements are created.

Input columns J:L for RAW, M:O divided by255 for JPG; output G:I is the original
measured XYZ scale. Do not apply gamma to RAW or convert JPG directly to sRGB
XYZ as if this were an absolute reference. Read only valid designated rows and
reject missing, formula, duplicate or nonfinite values. Ignore empty formatting.

Fit on FSCD200 only. Ordinary controls: constant training mean, linear3 terms,
affine4, degree2 polynomial10, degree3 polynomial20, homogeneous degree2 root
polynomial6, and a compact standard MLP(5,25,5). Root features are R,G,B,sqrt(RG),
sqrt(RB),sqrt(GB), with no intercept; this is an explicit variant, not an exact
reproduction of the author's seven-term basis. Linear systems use FP64 SVD least
squares. MLP uses train-fitted input/output standardization, tanh, sklearn L-BFGS,
alpha1, max_iter2000 and seed17. No hyperparameter sweep or target-based choice.
Retain convergence warnings and all methods. This MLP is not MATLAB trainbr.

5-fold GroupKFold by subject generates out-of-fold predictions on FSCD. Report
training-CV errors and select one reference control per format by lowest pooled
OOF XYZ RMSE. Refit all seven fixed controls on FSCD200. Serialize and hash all
14 models before reading Testing RGB/XYZ. Frozen risk score: root mean squared
distance to the five nearest training RGB vectors, in train-only standardized
RGB. All models share this deliberately simple score; it is an exploratory
support diagnostic, not a calibrated expected DeltaE or a learned innovation.

Evaluation: all100 testing sites and site-specific summaries, scalar XYZ RMSE
(over all3 coordinates), channel RMSE/MAE, median/p95 Euclidean XYZ difference,
and XYZ RMSE at100/95/90/80/70/60% retained sites. Break ties by SHA256 of stable
subject/site IDs. Report subject-clustered 2000-replicate bootstrap intervals
for selected-control XYZ RMSE, seed20260911. Selection uses only training OOF.

No DeltaE00 or DeltaE76: exact numeric reference white remains unverified. XYZ
error is a measured coordinate-space discrepancy, not a perceptual tolerance or
percentage skin-color accuracy. Do not convert it to degrees or a success claim.
This pilot does not evaluate any V2/V7 weight and must not inherit their claims.
