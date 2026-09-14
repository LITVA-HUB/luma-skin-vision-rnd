# Luma ChromaSeed-R v1 — prospective exploratory protocol

2026-09-13. Status: implementation before model fitting. This series continues the user's active autonomous research goal. No new confirmation set is available. The goal is not complete when this experiment or its software checks finish.

## Questions and controls

Can a compact model improve instrument-referenced skin color by (a) using local patch features, (b) applying a shared correction block repeatedly, and (c) selecting its active patch connections? Does inner-selected longer training improve generalization? The family name is **Luma ChromaSeed**; this iterative experimental branch is **ChromaSeed-R**. The name is provisional, with no trademark clearance claim.

Five separately trained families share a fit-only weighted ridge start (alpha 1): `stats_mlp`, `patch_mlp`, `recur_soft`, `recur_top16`, `recur_dynamic`. The ridge anchor is fitted to original 36 color features. Learned corrections are trained directly on fit rows; no prediction-error/confidence target or held-person residual head is fitted. Anchor-only performance is recorded as an additional reference. Historical KRR and compact-model results remain references; the new run does not replace them automatically.

- Stats: 36 → 96 → 96 → 48 → 3, SiLU hidden activations, tanh correction.
- Patches: shared 18 → 32 → 24 encoder for 64 cached patches; mean pool plus color36 → 96 → 64 → 48 → 3, tanh correction.
- Recurrence: same patch encoder, context60 → 48, shared state48. At each of four passes, query(state48, current Lab3) → 24 attends to projected patch24 keys. Updater(state48, context48, pooled24, current Lab3) → 48 → 48; next state is half the old state plus half tanh(update). Shared head48 → 3 adds 0.25*tanh(head) in normalized Lab. Thus the residual range over four passes matches the static tanh correction range. Initial head weights and bias are zero in all families.
- Soft uses all 64 patch connections. Top16 uses the largest 16 scores. Dynamic uses positive scores, with the top four always included; its training gate has a sigmoid straight-through derivative. Add 0.001 times mean sigmoid probability to the dynamic loss only. These are conditional connections, not newly created learned weights. Encoding/scoring is dense in this first implementation; sparse masks alone are not claimed as an acceleration.

The key projection is bias-free in all three recurrent controls. Synthetic engine-equivalence diagnostics exposed a mathematically redundant common key bias in soft/top16 attention: its near-zero FP32 gradient caused up to 0.0000193 weight drift, while non-bias weights differed by at most 0.000000187 and predictions by 0.00000763 native Lab after 64 steps. Removing that unused offset before real fitting avoids optimizing rounding noise. Dynamic uses the same bias-free projection for a consistent architecture (14,603 trainable values per recurrent model).

## Data and separation

Only the immutable original TRAIN cache with SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0` is permitted (966 images, 24 people). Load color36, tokens64×18, native Lab, and grouping metadata; do not load images or legacy validation/calibration/test. Reuse the exact historical mixed 18/6-person and both source-camera transfer roles, all labeled exploratory. Inner three-person-group folds use the frozen helpers from `skin_local_search_train.py`. All input/target normalization and the ridge anchor use only each fit fold. Sampling balances people, their sites, then images. No camera identifier is a model input.

## Frozen search and optimization

Three seeds 17/29/43 × three learning rates 0.0003/0.001/0.003 are trained as a GPU bank of **nine independent networks**, not an ensemble or shared optimizer state. Each slot has its own parameters, AdamW moments, LR and batches. LR variants for the same seed receive identical initialization and sample sequence. Batch size 64, constant LR, AdamW betas 0.9/0.999, epsilon 1e-8, weight decay 0.01, per-network gradient norm clip 5, deterministic FP32, TF32 disabled. Train 8192 updates and save checkpoints 512, 2048 and 8192. No schedule length effect contaminates early checkpoint comparisons.

CUDA execution uses a captured whole-step graph with a GPU update counter, precomputed identical sampling sequences and per-step Adam bias corrections. Three warmup steps are undone by restoring initial parameters, zero moments and zero counter before capture. Checkpoint weights must match eager execution in a synthetic equivalence test. CPU smoke retains eager execution. Capture/setup cost is included in training receipts. This implementation follows the constraints in [PyTorch 2.8 CUDA graph documentation](https://docs.pytorch.org/docs/2.8/notes/cuda.html#whole-network-capture).

Static loss is normalized Lab MSE. Recurrent loss is 0.6 times the mean MSE of all four predictions plus 0.4 times fourth-pass MSE. Query/update use the preceding prediction, allowing actual repeated correction. No palette pretraining is added to this series because v1 did not show robust transfer.

For each role and family, choose LR and checkpoint by mean person-balanced DeltaE00 from concatenated inner out-of-fold predictions, averaged over seeds; ties use smaller updates then LR. Select using the fourth-pass output for recurrence. Then lock choices before outer evaluation. This yields 405 inner training traces (45 banks), 45 final refits (15 banks), and 9 evaluated configurations per family/role. Checkpoints are correlated observations, not extra independent trials.

For recurrent models, additionally choose a convergence exit threshold in native Lab Euclidean distance from {0, 0.25, 0.5, 1.0}, minimum two passes, maximum four. Zero forces four passes. Choose fewest mean passes among candidates with inner person-mean DeltaE00 ≤ forced-four + 0.02 and image p90 ≤ forced-four + 0.1, averaged over seeds; ties prefer smaller threshold. This is a pragmatic exploratory tolerance, not proof of clinical or cosmetic equivalence. Store thresholds before evaluating outer rows. Report both forced-four and selected adaptive policies. Compute full traces for evaluation, but dedicated inference timing must actually break the loop for exited inputs.

## Evidence and resource accounting

Primary: person-balanced native DeltaE00; also image mean, site/person mean, p90, >5/>10, seed variability, per-person paired changes, actual iterations and active patch counts. Report quality at passes 1/2/4 without selecting from outer curves. Count learned parameters, anchor and normalizer bytes separately; report archive size separately. Time bank training, per-fit amortized cost, full search cost and batch-one CPU inference; do not present amortized bank time as standalone retraining latency. Feature extraction and face detection are outside model-only timing. Track peak allocated CUDA memory.

Scientific tests: bank AdamW against independent PyTorch optimizers, independent-slot training/forward equivalence, fold-only normalization/anchor behavior, gate cardinalities and nonzero gradients, actual adaptive exit parity with full traces. Freeze training/core/protocol/source hashes before real fits, verify immutable source hashes on resume, retain every negative result. Evaluation is a separate stage after selection; no outer labels influence fitting/selection. A runtime-only dry run may use generated arrays and may change batching implementation before the source lock, but must not tune using outer data.

The broader user goal still requires beating strong references with compact/fast inference and credible evidence across devices, followed by new representative phone facial data. Region-based exploratory accuracy is not validated end-to-end face analysis, identity recognition, retailer integration or evidence of a novel invention.

## Primary literature and provenance

Repeated computation and conditional execution are established research ideas, not novelty claims here: [Adaptive Computation Time, Graves](https://arxiv.org/abs/1603.08983), [Universal Transformers, Dehghani et al.](https://arxiv.org/abs/1807.03819), and [CondConv, Yang et al.](https://arxiv.org/abs/1904.04971). The proposed small tabular/patch regressor is our implementation for this specific exploratory experiment, not a reproduction of those papers. Existing dataset, code, and weight-license limitations remain in force.
