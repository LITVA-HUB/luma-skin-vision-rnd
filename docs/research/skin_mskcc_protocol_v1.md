# MSKCC direct skin-color pilot v1 — frozen before numerical endpoint analysis

Date: 2026-09-11. This is a NEW direct instrument-color experiment. Earlier
illuminant and He XYZ experiments remain separate, including negative results.

## Endpoint and provenance

Original dataset: https://api.isic-archive.com/doi/mskcc-skin-tone-labeling-dataset/
DOI 10.34970/962049, attribution Memorial Sloan Kettering Cancer Center.
The original landing page and every metadata image record specify CC-BY;
the landing page does not specify its version. No author neural weights or
third-party dermatology training corpora are adopted.

Use s7.csv normal-skin records joined exactly by isic_id to image metadata and
tag_id to s2.csv; patient_id must agree across the two joins. Exclude the three
literal not-available image IDs. Structural inspection before numeric analysis:
1838 images, 474 sites, 46 people (18 SLR, 28 iPod); 453 clinical close-ups,
1385 dermoscopic images. The full release has 4879 images / 64 people, not all
with instrument references. Never treat 1838 images as 1838 independent people.

Target: mean of the three supplied SkinColorCatch L*,a*,b* measurements at the
same site, verified against supplied average channels (rounding tolerance 0.051).
Missing/nonfinite references are rejected and counted, never imputed. Device
manufacturer specifies D65 / 10-degree observer:
https://store.delfintech.com/products/skincolorcatch . Predict native instrument
Lab directly; do not convert source image Lab to an assumed physical reference.
DeltaE00 compares predicted and measured Lab in this SAME instrument convention.
Image-derived Lab in s7 is a FEATURE, not a second ground-truth measurement.

Primary metric: per-image DeltaE00, kL=kC=kH=1; report mean, median, p90, p95,
fraction above 5 and 10, and patient/site counts. Also report patient-balanced
mean, modality and camera strata. DeltaE76 is secondary. Repeatability is the
pairwise DeltaE00 between three measurements at each unique source site; it is
a repeatability scale, not a certified instrument accuracy floor.

## Patient split fixed from identifiers, without inspecting color labels

Within each device sort patient IDs by SHA256("LumaMSKCCv1|" + patient_id).
SLR: first 4 TEST, next 3 VALIDATION, next 3 CALIBRATION, remaining 8 TRAIN.
iPod: first 6 TEST, next 3 VALIDATION, next 3 CALIBRATION, remaining 16 TRAIN.
All sites, modalities and repeated images of a person share the role.
No demographics, patient identifiers, site identity or device identity are
model inputs. Retain identifiers locally only for joins, grouping and audits.
Ten TEST participants remain numerically unopened until a separate model lock.

## First inexpensive falsifier: source-only image-summary controls

Use author-provided img_l,img_a,img_b only; no image pixels are required for
this tabulated pilot. It is NOT a locally reproduced image-to-Lab pipeline.
Fit TRAIN only and select on VALIDATION by patient-balanced mean DeltaE00.
Fixed candidates: constant mean; standardized affine ridge alpha 0.01/1/100;
standardized degree-2 polynomial ridge alpha 0.01/1/100;
standardized MLP(64,32), tanh, L-BFGS, alpha 0.1/1, max_iter 2000, seed17;
distance-weighted 5-nearest-neighbor regression. No additional candidates
after validation output in this version. Record all outcomes and warnings.

Report source repeatability using TRAIN sites only. For validation selection
curves use 5-nearest-TRAIN-neighbor standardized feature distance as a fixed
diagnostic score; this is NOT calibrated predicted DeltaE00. Report coverage
100/95/90/80/70/60 percent, accepting ceil(coverage*N) images with stable ID
tie-breaking. Do not fit any threshold or model using validation error for
claims of independent test performance.

CALIBRATION reserved for subsequent frozen error-head/post-hoc calibration;
TEST remains unopened during this first falsifier. Actual compact image neural
baseline, matched C+, proposed mechanism, camera-held-out tests, inference
measurements and ablations require separate frozen definitions and model lock.
Camera tests must train/validate/calibrate exclusively on the source camera;
the opposite camera's held-out people are then an unseen-camera test.

## Scope and limitations fixed before results

Source study: https://pmc.ncbi.nlm.nih.gov/articles/PMC12749783/ . Canon Rebel
T6i and iPod Touch 7 with Canfield attachment; study-initial white-balance
calibration; contact/non-contact and polarized/non-polarized imaging. Instrument
availability changed during recruitment. Device and subject distributions are
confounded; no paired capture of the same person across both devices is claimed.
Clinical close-ups are not ordinary uncalibrated facial selfies. This experiment
cannot establish iPhone/Android deployment accuracy or cosmetic shade matching.
The 2026 learned prior art https://arxiv.org/abs/2602.10265 predicts instrument
Lab for ITA with a different pretrained EfficientNet protocol. Published results
are not locally reproduced and direct Lab regression itself is not our novelty.
