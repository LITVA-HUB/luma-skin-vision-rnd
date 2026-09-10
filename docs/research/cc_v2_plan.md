# CC v2 improvement research plan — 2026-09-10

User authorizes sustained autonomous improvement and investigation. Preserve milestones2685bf0 and7637d6d, all negative evidence. No proprietary facial dataset, paid compute, external publishing, restricted data/weights or foundation-model inference. New branch codex/cc-equivariant-research.

## Design and scientific limits

The v1 direct CNN overfits the source sensor/illuminant distribution; this is a diagnosis to test, not an established causal finding. Compare an ordinary compact CNN with source augmentation against a positive-homogeneous anchor-normalized residual CNN. Let a(x) be channelwise GrayWorld/ShadesGray. Predict e(x)=normalize(a(x)*exp(r(x/a(x)))). For fixed valid pixels this construction is equivariant under positive diagonal gains. Reproduction error, unlike recovery angle, is invariant under jointly transformed prediction and reference. This wrapper is known prior art (Cotogni/Cusano2022/2024, FFCC/log-chroma predecessors); no novelty claim.

Source-normalized context plus relative hypothesis features may make the ERROR predictor more transferable. This is the empirical question. Compare standard context-only risk head with combined invariant features, fit using exact residuals on disjoint risk groups; source calibration and target evaluation remain separate.

## Global Constraints

- Read only source training/validation during estimator selection. Official462 test and Sony30 are previously observed regression sets, not fresh discovery evidence; do not tune against them. A genuinely fresh target-camera subset should be acquired independently if access permits.
- Existing cache/GT/metadata/split files are immutable and hashed. Real source supervision remains primary. Diagonal augmentation is labeled transformed-real augmentation, never new measured data; mathematical equivariance stress is a secondary diagnostic.
- Retain v1 and new negative variants. Fit model, selector and calibration on separate groups. No source-camera identity, CCM or target-camera statistics enter inference.
- Match backbone, resolution, optimizer, epochs and augmentation across direct/residual models. Use seeds17/29/43 for final matched contrasts; architecture/budget selected on source validation only.
- Measure full coverage, fixed coverage100/95/90/80/70/60, AURC/tails, frozen source thresholds, grouped bootstrap and full compute path. Physical ΔE00/skin accuracy remain unmeasured.

## Tasks

### Task 1: additive v2 experiment implementation

Implement tested homogeneous anchors, direct/anchored CNN variants, gain-consistent augmentation, immutable training/evaluation runner and invariant risk features. Exact brief: .superpowers/sdd/cc_v2_plan/task-1-brief.md. Preserve v1 code. Agent implementation and independent review before scientific runs.

### Task 2: source-only screening and matched experiments

Root executes source-only grid direct, anchoredGW, anchoredSoG; optional larger backbone if validation warrants. Equal120 epochs and augmentation. Select promising models using real source-validation reproduction mean; stress/gap is diagnostic, no test-based choice. Fit alternative risk heads with internal risk-group validation. Lock selected configuration before regression/fresh-camera evaluation. Additional seeds and ablations isolate mechanisms.

### Task 3: fresh camera evaluation

Investigate bounded HTTP-range access to legitimately licensed original/author-derived INTEL-TAU archives. Preselect deterministic camera-unique image IDs before reading pixels/GT, exclude previously used Sony examples and cross-camera shared scenes. The initial <=2GB per-camera target was expanded, within the user's authorized10–20GB initial budget, to128 images each from three cameras:4.434GB total transfer. Retain provenance/CRC/hashes, original BY-SA and preprocessing limitations. Acquisition succeeded before any target decoding or evaluation; details in docs/data/cc_v2_camera_acquisition.md.

### Task 4: evidence and decision

Independent review of algorithms/data/metrics; preserve failed candidates and source-only selection log. Export/profile only after positive measured evidence, include CPU preprocessing. Update results, status, prior art, dataset/IP and Skolkovo evidence. Finish with exact improvements versus v1 and strongest matched controls, limits, clean local commit and next research decision.
