# Frozen source representation screen: teacher versus absolute skin color

Frozen before skin teacher extraction or readout fitting. This screen tests
complementary information, not a deployable architecture or invented novelty.
Original MSKCC native-Lab TRAIN/VALIDATION roles only; no TEST/CAL or UMINHO
held-out access. All source camera families have already been explored.

## Feature extraction

Use already acquired original standard DINOv2 ViT-S/14, commit
7764ea0f912e53c92e82eb78a2a1631e92725fc8, Apache 2.0. Strictly verify original
source-file and weight hashes against the acquisition receipt. Load with
weights_only=True, strict=True; no network loading and no fine-tuning.
Teacher pretraining overlap cannot be independently audited.

Use existing original-image-derived 128x128 uint8 skin RGB caches only. No new
crop, segmentation or reference-based correction. Convert to RGB float/255,
CPU bilinear resize to 224 with align_corners=False, antialias=True, normalize
with mean [.485,.456,.406] and std [.229,.224,.225]. This upsampling adds no
new image detail. Frozen FP32 CUDA extraction, deterministic algorithms, batch
32, four CPU threads, xformers disabled. Concatenate the 384-dimensional
normalized CLS token and mean of 256 normalized patch tokens into 768 features.
Do not independently L2-normalize the two blocks. Replay all source features
using identical batching and require exact equality before readout fitting.

## Readout experiment

Six arms: color36; teacher768; color+teacher804; three color+teacher null
controls with teacher rows shuffled using seeds 17/29/43. Color features remain
unaltered in controls. Shuffle independently within TRAIN, selection and other-
camera evaluation subsets; for mixed fitting reuse selection features as
evaluation. A shuffle is not a new training seed or extra independent subject.

For each arm and each mixed / SLR-to-iPod / iPod-to-SLR protocol, fit six Ridge
alphas [.001,.01,.1,1,10,100]. Total 108 readout fits. No alpha refinement.
Standardize each feature column using fitting images only, replace std<1e-8
with 1, and divide each block by sqrt(block width). Concatenate blocks without
an additional dimension-dependent scale. Fit-only mean/std target scaling.
Fit sample weights give equal total weight per skin site, normalized to mean
one across fitting images. Model is linear Ridge with intercept and SVD solver.
CPU numerical routines limited to four threads. Retain every candidate.

Choose alpha using same-camera validation patient-balanced native DeltaE00;
break ties by declared alpha order. Other-camera source evaluation must not
choose any parameter. For mixed fitting selection=evaluation, explicitly
exploratory. Compare with existing strong compact capture plain/mixture and
training-only graph; do not call a teacher readout compact at inference.

## Metrics, controls and scope

Report single-image native-reference DeltaE00 mean/median/p95 and tail>5/>10,
patient/site counts, all source camera/capture strata and fixed coverage at
100/95/90/80/70/60%. Risk is mean distance to five nearest fitting inputs in
the same standardized readout representation, with no label access. This is
uncalibrated density, not expected DeltaE00 or a production rejection rule.
Write full risk curves. No ΔE from rendered spectra, RGB conversions or
illumination vectors; targets are actual native instrument Lab.

Require saved coefficient replay, independent scalar metrics and independent
weighted normal-equation checks. Verify all scalers and sample weights use
fitting data only, every chosen alpha follows selection only, and all shuffles
preserve exactly their source subset. Keep arrays/weights/IDs outside Git.
Report the full teacher parameter/file cost, not just tiny readout coefficients.

Decision: if teacher adds no consistent advantage over color and shuffled
controls, do not begin a major distillation run on this representation. Even
a source gain would require a compact-student test and future independent
reference validation; it would not establish ordinary facial-phone accuracy.
