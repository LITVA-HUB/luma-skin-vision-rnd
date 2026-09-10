# V7: sensor-response augmentation and privileged canonical scene supervision

User steering: camera independence through scene/physics, not device identity.
V6 canonical-frame fusion is negative so far; do not assume it is the solution.
Keep the Fourier/decision-set branch alive as an independent mechanism.

Hypothesis: a compact model can learn scene features that survive changes in
sensor response if a frozen semantic teacher supervises a photometrically
canonical training view. The teacher and real GT are training-only. At inference
there is one128px image, no camera name, CCM, teacher, adaptation or extra image.
A canonical teacher target is an UNVERIFIED contribution candidate. Generic
KD, clean-target feature distillation and semantic color constancy already exist.

Pre-extraction guard caught a protocol mismatch: 38 of 44 TRAIN dates also
occur in the official SimpleCube TEST split. TRAIN is date-disjoint from VAL,
RISK and CAL. The official TEST is image-disjoint, not date-disjoint; retaining
the established 1126-row training population is deliberate. The initial cache
attempt stopped before extracting images/GT/features or creating any output.
An explicit exception accepts only rows marked both subset=test and
official_split=test; all fitting-role overlap still fails. Per-role overlap is
saved in the cache audit. No official TEST input/GT is decoded in this screen.
External camera evaluation remains separate. Do not claim universal capture
group independence for the historical official SimpleCube test protocol.

Fixed source-development screen, seed17 first: five120-epoch arms, batch32,
AdamW lr.001/decay.0001, cosine decay to.00002; same1126 train/119 val, from
scratch, matched initialization/ordering/exposure/flip RNG. All GT real-source
illumination labels. Variants: gt_native, gt_sensor, raw_teacher_sensor (C+),
canonical_teacher_sensor (candidate), canonical_teacher_native (ablation).
Same MobileNetV3-large point/context modules as V2 direct; a960->384 projection
supervises4x4 features only during training and is removed for deployment.
Training loss: reproduction-angle GT loss plus mean1-cosine patch distillation
with coefficient1 for teacher arms. Native validation mean selects checkpoints;
no selection on transformed images or observed phone tests. Preserve best/final.

Sensor augmentation: M=D*((1-epsilon)I+epsilon*A), A nonnegative row-stochastic,
epsilon uniform[0,.35], diagonal D gains exp(uniform[-.7,.7]);25% identities.
Apply the SAME M to linear pixels and GT; normalize GT, no pixel clipping. In a
linear spectral sensor model this equals a nonnegative combination of original
sensor sensitivities. It is physically justified within that restricted span;
it does NOT synthesize arbitrary cameras or nonlinear ISP/HEIC pipelines. Real
scene images/labels anchor training. These virtual sensors are augmentation,
not new instrument measurements and not real unseen-camera evaluation.

Teacher: already acquired standard DINOv2-S/14, Apache2.0, source/weight SHA
verified from original Meta release. No XRay/Cell weights. Extract TRAIN ONLY:
raw and GT-diagonally-corrected camera RGB views, each original/flipped. Normalize
common exposure with95th percentile, clip to[0,1], gamma1/2.2, resize224; apply
ImageNet channel normalization. This is a declared neural rendering convention,
not a colorimetrically calibrated sRGB image. Pool256 teacher patch tokens to4x4
and normalize per token. Raw-teacher C+ receives identical extra pretraining,
projection, compute/training budget and sensor augmentation. GT-based teacher
correction is the sole proposed difference against this strongest matched C+.
Teacher pretraining benchmark overlap remains unknown and must be disclosed.

Cache only1126 TRAIN feature rows, with source-role/data/teacher/render hashes.
Never extract teacher test/phone features. Validate augmentation algebra with
independent spectral integration; check positivity, identity and reproducibility.
A held-out virtual-sensor stress test may diagnose behavior, but cannot replace
newly locked real-camera testing. Do not change camera/refusal claims from a
source or virtual-sensor result alone. Risk calibration comes after choosing a
source-only estimator; keep ordinary matched risk heads as controls.

Mechanism failure modes: canonical RGB is still source-sensor-dependent;
semantic features may remove useful chromatic evidence; teacher targets may be
misaligned; sensor span may fail to cover real devices. A first-seed win only
justifies further seeds and a fresh fixed external test, not novelty/SOTA.
