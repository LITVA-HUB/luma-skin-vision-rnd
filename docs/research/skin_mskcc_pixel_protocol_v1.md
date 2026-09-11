# MSKCC pixel experiments v1 — source-development protocol

Frozen before local pixel feature results or image-model training. Date2026-09-11.
Preserve original subject roles and data/protocol hashes from skin_mskcc_protocol_v1.
TRAIN966images/24people, VALIDATION264images/6people only. Calibration208images/
6people and TEST400images/10people remain unopened. Source validation is already
used for research selection; no independent efficacy claim may use it.

## Pixels and labels

Original CC-BY JPEGs checked against acquisition SHA256 before decoding.
Pillow EXIF transpose, RGB bytes, central square with side floor(0.8*min(W,H)),
resize128x128 with LANCZOS. No automatic white balance or ICC conversion.
These are encoded camera pixels, not asserted physical linear RGB/sRGB colors.
Targets are original mean triplicate instrument Lab, in its native convention.
No camera, patient, site, demographic, author-derived image Lab or measurement
replicate variance enters inference. Site identity is used only for audits.

## Fixed inexpensive representations and controls

Locally extracted median RGB(3), color quantiles/mean/std/correlation(36),
joint RGB histogram8x8x8(512), and concatenated color+histogram(548).
Quantiles are .01,.05,.1,.25,.5,.75,.9,.95,.99 for each RGB channel; three
off-diagonal covariances divided by per-channel std products, zero if constant.
For each representation: standardized ridge(alpha1), polynomial2ridge(alpha1)
only for median, standardized target/input MLP(64,32),tanh,L-BFGS,alpha1,
max_iter2000,seed17. Nine controls total. Predict native Lab directly. No fit
on validation. Each model's fixed density score is distance to5TRAIN neighbors.

## Image models: one standard and one precisely matched mechanism probe

1. Standard torchvision MobileNetV3-small, weights=None,num_classes3. Standard
   architecture baseline, no external pretraining. Torchvision0.23.0/BSD-3-Clause.
2. Compact patch-vote network: each of64nonoverlapping16x16patches supplies RGB
   .1/.5/.9 quantiles, mean, std and per-channel mean absolute spatial gradient
   (18features). Shared MLP18->256->256 with SiLU; pooled mean/max/std(768) goes
   through context MLP768->512->512. Per-patch [local256,context512] ->256->4
   gives standardized Lab vote(3) and softmax confidence(1). Confidence-weighted
   sum is the baseline. Camera/pixel position is not a feature.
3. Identical architecture/initialization/data/budget, with3differentiable Huber
   reweighting updates. Starting at the confidence-weighted sum, multiply the
   original confidence by min(1,5/max(physical Lab Euclidean residual,1e-6)),
   renormalize and recompute. Delta5 is an INTERNAL DeltaE76 scale, not DeltaE00.
   This tests robust iterative aggregation, not a new physical law or established
   novelty. Compare1and3steps at evaluation, but choose training arms in advance.

Patch summaries remove spatial arrangement but retain absolute color. Hypothesis:
multiple local color witnesses let a small model suppress hairs/glare without
estimating a separate global illuminant. Fundamental risk: every local vote may
learn the same global shortcut, providing no independent evidence. Also real
skin variations are not necessarily outliers. Retain vote dispersion diagnostics.
Deep Sets, confidence aggregation and Huber estimation are established mechanisms;
implementation alone is not novel. This is NOT yet full calibrated C+: a standard
error head/post-hoc calibration remains a subsequent matched phase with held-out
source residuals. Do not label density/vote disagreement as calibrated error.

## Fixed training recipe

All3architectures, seeds17/29/43;80epochs,AdamW,lr0.001,weight_decay0.01,
cosine schedule to0.00001,batch32. Float32,RTX4060, no paid compute. Target
standardization fitted on TRAIN only; squared standardized Lab loss. Random
horizontal/vertical flips for CNN only, label-preserving; patch-set summaries
are invariant to these transforms up to token permutation. Same deterministic
shuffle order by seed/epoch for both vote arms. All TRAIN images occur once per
epoch. No skin tone class weighting or test-camera conditioning.

Select epoch by VALIDATION patient-balanced meanDeltaE00, not training loss.
Report each seed plus arithmetic mean prediction across seeds separately.
Retain best/final checkpoints, histories, warnings, parameter count, peak GPU
allocation and all predictions. Batch1 timing is measured only for completed
candidate inference; preprocessing timing must be distinguished.

Primary development endpoint: actual DeltaE00 mean,median,p90,p95,above5/10;
patient-balanced mean; clinical versus dermoscopic and camera strata. These
camera strata are KNOWN cameras. Unseen-camera runs require separate source-
camera-only fits/validation/calibration and a later frozen evaluation protocol.
No independent test, deployment or universal skin accuracy claim is permitted
from this development screen. No prediction-dependent data exclusion.
