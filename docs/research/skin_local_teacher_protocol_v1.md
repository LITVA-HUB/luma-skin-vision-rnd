# Local teacher / RGB correspondence source screen

Frozen before local-token extraction and fitting. This is a teacher-assisted
readout experiment, not a deployable compact student or an invention claim.
Original MSKCC native Lab TRAIN/VALIDATION only. No reserved TEST/CAL or UMINHO
held-out access. Camera-source evaluation uses previously explored cohorts.

Use unchanged licensed DINOv2 ViT-S/14, original source/weight checks and
preprocessing from skin_teacher_features.py: existing 128px RGB upsampled to
224px, FP32, deterministic CUDA, batch32. Teacher remains frozen. Its 16x16
patch grid is pooled in exact non-overlapping 2x2 blocks into an 8x8 row-major
grid, geometrically corresponding to the existing 64 RGB patch summaries.
Teacher context can span the whole image, so a token is not strictly local.
Check all source features by exact replay and independently loop-check pooling.

Use the strong existing CaptureColor('mixture') as the base, initialized first
with the same seed and unchanged dimensions/auxiliary mode loss. Add one
adapter after the base modules: Linear(384,128), SiLU, Linear(128,256).
After the base local RGB encoder, add 0.5*tanh(adapter(normalized teacher)).
Teacher channels are normalized using the fitting subset's per-channel mean
and std over images and patches; std<1e-8 becomes1. Never fit this on selection
or other-camera samples. Original absolute RGB summaries remain unchanged.

Four arms have identical stored shapes and initial weights:
- plain: adapter disabled, exact original mixture control; no teacher needed;
- aligned: teacher descriptor of the same spatial cell;
- global: image-average descriptor broadcast to every cell;
- shuffled: fixed deterministic within-image permutation, changing descriptor/
  RGB correspondence while preserving all descriptors from that image.

Shuffled permutations derive from SHA256('LumaLocalTeacher1|'+image identifier),
first8 bytes interpreted little-endian as NumPy default_rng seed; 64-token
permutation generated once per image. Identifiers/permutations stay outside Git.
This is a diagnostic control; the aligned model requires no image identifier.
Global descriptors are averaged before fixed affine normalization.

36 fits: four arms x seeds17/29/43 x mixed / SLR-to-iPod / iPod-to-SLR. Unchanged
80epoch AdamW, learning rate .001, weight decay .01, cosine to .00001, 16 paired
sites per32-image step, ceil(n/32) steps/epoch. Standardized-Lab MSE plus .1
capture-mode cross-entropy as in historical mixture. Mode/camera/reference is
not an inference input. Target scales fit on the fitting subset only. Epoch
chosen by same-camera validation patient-balanced native DeltaE00. All final
other-camera results retained; no target-camera parameter choice.

All output metrics use one image and genuine native Lab: full mean/median/p95,
tail>5/>10, 100/95/90/80/70/60% coverage and full risk curves. Risk is the same
uncalibrated four-hypothesis RMS Lab dispersion across arms, not calibrated
expected DeltaE00. Preserve strong historical plain/mixture/training-only graph
comparators. Report teacher cost separately and together with the trained head;
precomputed features do not eliminate the teacher's inference requirement.

Main falsifier: aligned must improve over global and shuffled correspondences,
and over strong ordinary compact controls, beyond a single protocol. A positive
teacher-assisted result is only a gate for future student/distillation work;
no claim that a 1M student already achieves the teacher-assisted accuracy.
No fine coefficient search, deployment optimization, or new independent-test
claim in this screen. Same-camera people differ from other-camera people, so
observed transfer is not pure camera causality. Pretraining overlap is unknown.
