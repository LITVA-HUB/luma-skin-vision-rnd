# Capture-aware color and perceptual-objective factorial experiment

Previous turn was PROGRESS:27fits rejected unconditional capture invariance.
Invert that assumption: retain capture-process evidence and infer it from the
single image. Keep actual instrument-referenced skin color as the objective.
No original CALIBRATION/TEST cache, labels or predictions are opened here.
Only existing TRAIN24people/966images and VALIDATION6/264 are used; source
validation has prior research exposure, so all new results are exploratory.

## Hypotheses and exact controls

All architectures have the same929,297stored parameters and same initialization
per seed: original18->256->256patch encoder; mean/max/std context768->512->512;
per-patch [local256,context512]->256->13 head (four3Dcolor hypotheses and one
shared patch-confidence logit); context512->4 capture-mode gate. The gate is
predicted from the image; neither camera ID nor capture mode is an input.

`plain`: uniform average of four hypotheses, no mode auxiliary loss.
`uniform`: same uniform color average, plus0.1cross-entropy for capture mode.
`mixture`: gate-weighted hypotheses, same0.1capture-mode auxiliary loss.
The strict matched baseline for mixture is uniform: exact weights/capacity,
data, mode supervision, optimization budget and objective. Plain isolates the
value of auxiliary supervision. Hypothesis logits all receive gradients through
the shared soft mixture; no externally provided mode or oracle gate is used.
Modes in fixed order: NP-C, NP-NC, P-C, P-NC (original acquisition annotations).
Without auxiliary loss the gate is unused for color and its accuracy is merely
an untrained diagnostic. Uniform gate is training-only for color prediction;
report stored versus functionally used components honestly.

Mechanism: visually distinguish capture effects before combining color estimates.
Assumption: process appearance is recognizable and useful for predicting true
color. Failure: gate learns device/person shortcuts, hypotheses collapse, or
nominal capture modes do not capture the important variability. Matched uniform
baseline and held-out-camera fitting are the cheapest falsifiers. Mixture of
experts and auxiliary classification themselves are NOT novel principles.

Each architecture is trained under both standardized-LabMSE and true squared
CIEDE2000/25. This yields six fixed configurations. Direct perceptual geometry
may align optimization with reported error; it can also overemphasize tails or
become unstable. It is an objective control, not a novelty claim. CIEDE2000 uses
native instrument Lab, never angular proxies or reconstructed sRGB labels.

## Numerical loss validation

Torch implementation follows the already independently audited formula, computed
in float64, kL=kC=kH=1. All34Sharma reference pairs must agree within5e-5DeltaE00;
finite-difference gradients must agree away from discontinuities; exact neutral
and equal colors must have finite gradients. At a zero square-root argument
define zero derivative and use clamp1e-24 in the evaluated positive branch.
At neutral atan2 use the zero-hue convention with safe arguments. This does not
make the original hue-boundary formula globally differentiable. Native training
references remain float64 for the perceptual loss. Evaluation uses the frozen
independent NumPy/scalar CIEDE2000 implementation.
[Original implementation notes and discontinuities](https://doi.org/10.1002/col.20070).

## Fixed runs before new results

Full factorial:3architectures x2objectives x3seeds17/29/43 x3protocols =54fits.
All80epochs, AdamWlr.001/wd.01, cosine end.00001. Same actual same-site paired
sampling as previous screen (16sites,2views per step; ceil(ntrain/32)steps/epoch),
but NO consistency penalty. Check finite loss/gradient norm; no gradient clipping
(max_norm=infinity). Only color loss changes between objectives. Inference always
takes a single prepared64x18token image. No pretrained weights or synthetic
augmentation. Standardization uses only fitting-person native targets.

Mixed: fit24TRAIN people, epoch-select on6VALIDATION people.
FromSLR: fit8TRAIN SLRpeople/323images, epoch-select3VALIDATION SLRpeople/132images;
evaluate3VALIDATION iPodpeople/132images only after best epoch chosen.
FromiPod: fit16TRAIN iPodpeople/643images, select3VALIDATION iPodpeople/132images;
evaluate3VALIDATION SLRpeople/132images after selection.
All54configurations are specified now; there is no source-selected challenger
subset for camera fitting. No opposite-camera normalization, auxiliary labels,
images or color targets enter a fit or its checkpoint selection. Architecture
research has seen both cameras historically: this is source-only exploratory
camera-held-out fitting, not independent final confirmation or ordinary selfies.

Keep best/final checkpoints, histories, native predictions and all failed arms.
Report mean/median/p95/tails, per-mode/camera/type, mode accuracy/entropy and
between-hypothesis dispersion as diagnostics. Mean of seed scores is not ensemble
accuracy. Do not tune on this factorial's camera endpoints. Independent replay
must check target scales, initial states, all checkpoints and scalar DeltaE00.
Only after a reproducible accuracy benefit should new selective calibration or
deployment optimization be considered. Prior independent skin evidence and all
synthetic/angular negatives remain intact; universal facial-phone accuracy and
the product median<=2/p95<=5 at80% acceptance remain unmet.
