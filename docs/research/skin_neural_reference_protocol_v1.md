# Frozen compact neural-reference skin adapter screen

Question: can a strong fixed neural representation plus query-dependent local
residual regression beat the same core and an ordinary equal-capacity head?
Actual instrument-native MSKCC skin Lab / CIEDE2000, original CC-BY release.
No new datasets/weights, independent TEST/CAL, ordinary-phone claim or cloud.

## Source roles and historical exposure

Use original TRAIN24 people/966 photographs and VALIDATION6/264. Banks: mixed,
SLR-only8/323, iPod-only16/643. Evaluate mixed known cameras, and3 known plus3
unseen people per single-camera bank. Camera and person/capture are confounded.
Source validation has already been extensively explored. All core checkpoints
are existing skin_capture_v1 mixture_mse best.pt seeds17,29,43, selected using
their corresponding known-camera source validation. They exclude unseen camera
training/selection but are not fresh independent checkpoints. Freeze their
hashes before fitting. No source validation selects adapter epochs/configs.

## Three matched trainable adapters and unchanged-core control

Core929,297 parameters stays frozen and identical within each protocol/seed.
Cache512 pooled learned context,36 image color descriptors and3 standardized
base Lab predictions. Standardize context/descriptors with TRAIN bank only;
base predicted Lab is already in TRAIN target coordinates. Total input551.
Head551->256 SiLU ->192 SiLU ->16 tanh ->3. All heads193,795 parameters,
total1,123,092 <= user cap1,129,297. Same random initialization; final3 outputs
zero-initialized, all initial corrections exactly zero. No extra learned scale.

C+: ordinary tanh3 residual added to standardized base prediction.
Reference mean: the same16-dimensional head embedding builds Gaussian weights
exp(-0.5*mean squared embedding distance), masking all same-person bank images.
Reference affine: identical weights, but local affine fit in16 embedding
dimensions to bank residuals (standardized reference Lab minus fixed base).
Weights sum1; unpenalized weighted intercept, ridge penalty0.01 on slopes.
Evaluate the local residual at the query embedding; multiply each correction
coordinate by tanh3 gate. Mean arm supplies the averaging ablation.

Training queries exclude ALL images of their person from the reference bank.
The frozen core saw TRAIN people: this is ordinary training, NOT OOF accuracy
or OOF uncertainty supervision. Bank residuals are in-sample core residuals;
their optimism and acquisition bias are explicit failure modes. Validation
people/sites are excluded from core training and every reference bank. Inference
requires one image and fixed TRAIN memory, no camera ID or query ground truth.

## Fixed optimization and reporting

27 adapter fits:3 protocols x3 seeds x3 arms.300 AdamW steps each, batch32
uniform TRAIN image draws with replacement, identical draws across arms for a
given seed/protocol. Learning rate0.001, weight decay0.01, cosine to0.00001.
Standardized Lab MSE objective. Core remains frozen; no mode loss or metadata
input. Final checkpoint only; no stopping/selection based on validation. Report
all arms/seeds plus unchanged cores; aggregate individual errors, no ensemble.
Fixed reference weights/solver settings; no result-driven configuration edits.

Report mean/median/p95/>10 and person/site-balanced native DeltaE00. Identical
nearest-TRAIN standardized36-descriptor risk order for every arm:100/95/90/80/
70/60% and full curves. This is a diagnostic common ranking, NOT calibrated
expected error, C+ confidence head or reliability guarantee. Calibration would
require a separately frozen person-excluded residual procedure if color helps.

Measure allocated CUDA peak for cached head training and batch1 neural+adapter
inference on prepared tokens/descriptors. Include frozen-core memory; explicitly
exclude JPEG decode/feature extraction. Save deployable core/head/scales plus
cached16-embedding/3-residual bank; report bank scalars and complete file bytes.
Private training caches/person IDs stay ignored and are not deployment inputs.

## Prior art, falsifiers and next decision

[MetaOptNet,CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html)
learns embeddings through convex base learners;
[Bertinetto et al.,ICLR2019](https://arxiv.org/abs/1805.08136) differentiates
through closed-form ridge solvers. This experiment is not their reproduction;
no author code/weights/numbers are imported. Learned kernels, local regression,
residual learning and gating are known; their combination alone is not novelty.

Main assumption to attack: residuals near a query transfer across people and
cameras. Inversion: let an ordinary head absorb the mapping without reference
memory (C+). Mean control tests whether solving a local affine system is needed.
Failure modes: optimistic TRAIN residuals, wrong neighborhoods, acquisition
bias, insufficient support, overfit gate and unstable extrapolation. Numeric
tests compare independent augmented least squares and finite differences,
excluded-person perturbations, constant limits and equal initial predictions.
If both reference arms lose C+, do not enlarge/sweep the bank blindly. If local
affine beats mean and C+ in both directions, freeze a follow-up with properly
person-excluded encoders/residuals and a separate0/1/2/3-pass refinement ablation.
This is one exploratory screen; independent victory remains unproven.
