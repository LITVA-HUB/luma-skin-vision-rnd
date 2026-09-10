# V3 architecture screen: full color-frame posterior graph

**REAL development-validation result: the proposed full-frame architecture loses.** After equal120epoch training, its mean reproduction error is4.281° versus2.547° for the same graph operating in ordinary RGB. At nominal80% coverage the errors are4.155° versus2.304°. Preserve this negative result. This is a source architecture screen, not a new independent benchmark milestone or a camera-generalization claim.

## Architecture implemented

The network selects a three-vector color basis from16 spatial patch means, solves for pixels in that basis, creates64 relational tokens, and processes them with four edge-gated graph diffusion blocks. It predicts a point illuminant and eight directional posterior components, then transports posterior quadrature hypotheses back to camera RGB to compute an approximate reproduction-risk score. It has no CNN backbone, imported weights, foundation model, camera identity or CCM input.

Direct, diagonal-GW and full-frame variants have identical graph/heads and exactly1,215,535 trainable parameters. The full-frame algebra has known canonicalization/frame ancestry; implementation is not proof of scientific novelty. Jointly admissible constructed transformations verify the projective transport identity. Numerical rank, conditioning, selection ties and positivity restrict that identity. Full reproduction-risk geometry is not invariant to arbitrary channel mixing.

## Real data, protocol and rights

SimpleCube++ provides real illuminant ground truth under original CC BY4.0 terms. Only1,126 training and119 development-validation images were decoded; their capture-date groups are disjoint. The official462 test images,259 risk-fitting images,268 calibration images and all INTEL-TAU labels remained outside this estimator screen. This119-image validation set has been used in earlier development and selected checkpoints; the numbers below are not independent test estimates.

All three modes receive seed17,128×128 thumbnails,120epochs, batch32, AdamW0.001, matching cosine schedule, augmentation and loss weights. Exact choices preceded learning in [the screen lock](../research/cc_v3_source_screen_lock.md). Checkpoint selection minimizes validation reproduction mean among epochs with at least99% supported predictions, including untrained epoch0. Selected checkpoints all support119/119 images. Some later full-frame epochs refuse one image; these failures remain in the full history.

Each run saves exact executable snapshots, data hashes, training/validation IDs, every epoch, best prediction arrays and checkpoint hashes. [The report generator](../../scripts/cc_v3_source_report.py) independently recomputes angular errors with atan2, checks summary defects≤1e-6°, validates best-epoch selection and confirms identical data/code/budgets. Evidence: [summary JSON](cc_v3/source_screen/summary.json), [learning/risk curves](cc_v3/source_screen/source_screen.png). The declared two-epoch smoke is separate and has an explicit [epoch0 reporting erratum](cc_v3/feasibility_erratum.md).

## Measured source-validation results

All angles are degrees; lower is better. These three models are REPRODUCED LOCALLY.

| Graph coordinates | Best epoch | Recovery mean | Reproduction mean | Reproduction median | Worst25% mean | p95 | Error>10° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct RGB, matched control |29|1.9043|2.5465|1.0773|7.1050|8.9890|3/119|
| Diagonal GW |69|2.2387|2.9416|1.9369|7.0077|8.0856|5/119|
| Full color frame, Proposed |54|3.1653|4.2813|3.2137|9.8680|11.1996|10/119|

Previously reproduced classical estimators on the same validation IDs give mean reproduction4.4669° for Gray World and4.2479° for Shades of Gray. Classical features use the original full-image preprocessing, whereas neural inputs are128px thumbnails; do not call them resolution-matched. Full-frame Proposed also fails to beat this Shades of Gray result. The prior3.034M direct CNN seed17 reached2.4141° on this development validation; the new direct graph is smaller but has not beaten that stronger historical learned reference. No author-published numbers are presented as locally reproduced.

## Uncalibrated selective diagnostic

The ranking score is the approximate transported posterior spread. No held-out error head or post-hoc calibration was fitted in this screen. Both checkpoint selection and these diagnostics use the same development validation, so they cannot establish calibrated reliability.

| Nominal coverage | Direct RGB | Diagonal GW | Full frame Proposed |
| ---: | ---: | ---: | ---: |
|100%|2.5465|2.9416|4.2813|
|95%|2.5337|3.0426|4.2065|
|90%|2.4094|3.1677|4.1644|
|80%|2.3040|3.1668|4.1549|
|70%|2.0776|3.0870|3.8025|
|60%|1.8577|3.2247|3.6498|

Counts use floor(119×coverage), matching the existing evaluator; nominal80% is95/119=79.832%. Full119-point curves, tail summaries and tie order are in JSON. **Diagonal posterior ranking increases error as coverage is reduced**, a second negative finding. A directional posterior by itself does not guarantee useful error ordering. Low-order quadrature has measured orientation artifacts; its invalid-mass output is an approximation, not an exact probability or certificate.

## Hardware and feasibility

All three models:1,215,535 parameters;4,905,818-byte best checkpoint; peak PyTorch allocated training643.09MiB including233.45MiB cached source data on RTX4060 8GB. The full120epoch loops take114.78s direct,116.49s diagonal and123.26s frame on this host. These are local loop timings, not inference measurements; OS/driver/CUDA context memory is outside the allocator measurement. Concurrent CPU verification may affect timing.

V3 batch1 latency, inference VRAM, ONNX and TensorRT are **NOT MEASURED**. Do not optimize/export this losing full-frame candidate merely to produce deployment numbers. V2's separately measured deployment artifact remains preserved.

## Separate next-camera population

Acquired384 previously unused INTEL-TAU images,128 per Canon5DSR/NikonD810/SonyIMX135, with4,436,408,351 bytes transferred. All768 image/reference files verify against pinned member metadata. Original CC BY-SA4.0 controls use; the mirror's MIT badge is ignored. This new V3 research training track is separately documented and is not blanket clearance for unrestricted proprietary model distribution. No V3 weights have yet been trained on INTEL-TAU.

There is zero old/new image-path or image-hash overlap, but67 new rows share exact reference-file hashes with V2. [The grouping audit](../data/cc_v3_grouping_audit.md) records this before any new target error inspection. Reference hashes are only proxies for scene groups. New LOCO roles and near-duplicate checks are pending; unseen-camera V3 results remain **NOT MEASURED**. Pinned mirror membership/integrity does not prove whole-byte identity to original publisher archives.

## Decision and limits for Luma

Reject the initial full-frame architecture as the current quality candidate. Its strict invariance and canonical metric may discard useful color priors or make learning difficult. Those are supported explanations to investigate, not separately proven causal findings. Follow [the revision decision](../research/cc_v3_revision_decision.md), preserving these runs.

This work establishes implementation, numerical limits, hardware feasibility and a real-data negative architecture result for the future photometric-normalization component. It establishes no new positive benchmark superiority, facial skin accuracy, CIELAB/ΔE00 accuracy, guaranteed selective risk, universal camera independence, patentability or legal Skolkovo classification. The positive but limited V2 camera-transfer evidence remains in [its original report](cc_v2_report.md).
