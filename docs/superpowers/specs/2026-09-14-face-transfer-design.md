# Seg2: controlled continuation of facial skin segmentation

The user explicitly authorizes autonomous experiments, additional training steps and use of suitable public datasets. This design selects a bounded scientific intervention within that authorization. It does not modify any frozen Seg1, HR, P1, P2 or P3 file. No delegation; work in the existing `codex/skin-local-search-2026-09-13` worktree.

## Question and choice

Does adding the cleaned CelebAMask-HQ TRAIN improve the existing useful Seg1 model compared with spending the same additional updates on LaPa alone? Does extending each trajectory improve its validation-selected result?

Compare two continuations from the exact same Seg1 checkpoint. A fresh-from-scratch comparison would discard the existing learned representation and add training cost; changing architecture at the same time would confound the data intervention. Seg2 retains the 4,416,673-parameter width-24 SkinUNet, input 192×192, GroupNorm, SiLU and logit-zero mask threshold. The old optimizer state was not saved, so these are fine-tuning runs with a fresh AdamW optimizer, not exact continuation of old optimizer dynamics.

Initial checkpoint: `D:/Luma-RnD/data_growth_2026_09_14/facial_skin_v1/best.pt`, SHA256 `1203cbc5ed2ee17cb2a408c47a23b3a28468f169d48b4d3cee8bd3059e7fbed3`. Verify its existing selection/protocol/seal bindings before use.

## Fixed comparison

- Arms: `lapa_only` and `lapa_celeba`; seeds 17, 29, 43, six trajectories.
- Batch 32. Each batch has a common 16-image LaPa anchor and 16 additional images. The additional stream uses LaPa for `lapa_only`, CelebAMask-HQ for `lapa_celeba`.
- Anchor samples and augmentation random numbers are matched across arms at a given seed/step. Additional streams have separate deterministic generators. Sampling traverses independently shuffled complete allowed-index lists and reshuffles at exhaustion. Cross-stream duplicate LaPa indices are permitted and counted; no sampler may draw from an unfiltered raw pool.
- Clean LaPa TRAIN 15,914 / VALIDATION 1,692; old TEST 2,000 already exposed. CelebAMask-HQ TRAIN 24,112 / VALIDATION 2,992 / TEST 2,822 from `celeba_mask_hq/split_v1`. Never use the 30,000-row raw pool as a training split. Provided person codes, exact-byte groups and components are disjoint within the cleaned CelebA splits. Cross-source same-person independence is unverified.
- Fixed reference interval 498 updates (`ceil(15914/32)`). Validation at step 0 and every 498 updates. Budget windows 1,494 / 2,988 / 5,976 updates (3 / 6 / 12 reference intervals). The same trajectory and learning-rate horizon serve all prefix comparisons.
- Fresh AdamW, learning rate 0.0001, weight decay 0.0001, cosine multiplier 1.0 to 0.1 over 5,976 updates. BCEWithLogits plus per-image soft Dice, gradient norm clip 5. Reuse frozen Seg1 augmentation: flip .5, channel gain .9–1.1, exposure/gamma .85–1.15.
- CUDA production uses channels-last and FP16 autocast/GradScaler with finite-loss and finite-gradient checks. Report attempted batches and successful updates; nonfinite gradients fail the run instead of silently consuming a budget step. Fixed seeds do not establish bitwise cross-device reproducibility.
- A full trajectory sees 191,232 extra image presentations. The mixed arm sees 95,616 per source; reference intervals are not full passes through the combined dataset. Total six-run upper budget 35,856 updates / 1,147,392 image presentations.

## Selection and evaluation

Evaluate full LaPa and CelebA validation separately at every checkpoint. Selection score is the unweighted mean of the two source mean-per-image IoUs. For each arm/seed/budget, select the highest score at or before the budget, earliest step on a tie. Include the unchanged step-0 checkpoint; a worsening fine-tune must be allowed to lose to its initialization. Freeze all 18 arm/seed/budget choices and the overall deployment choice before test inference. Overall choice uses full-budget validation score, then lower selected step, arm order, seed order as tie breakers.

Report all six trajectories, all budget windows and negative outcomes. Main data-effect comparison is paired by seed at the same budget, source-wise and macro averaged. Do not select the best test run. Report the standalone unchanged Seg1 on both sources with the same evaluator after choices freeze.

Held evaluation reports per-image/global IoU, Dice, precision, recall, the fraction of images below IoU .5, and counts. Keep old LaPa TEST labeled exposed. New CelebA TEST has had technical data/mask integrity checks but no model inference. Preserve explicit source and grouping limits.

Secondary color extraction metric uses the existing frozen `skin_face_appearance.appearance` / `color_error`: mean and median RGB colors inside the manual and predicted masks on the same 192×192 image, assumed sRGB/D65/2°. Minimum 16 mask pixels; report missing predictions and common coverage. This is apparent color extraction error, not instrument Lab accuracy. Never add these labels to the native color regressor.

No new network layers are added, so count all parameters and saved model bytes. Measure actual CPU inference only after training/verification GPU jobs are finished and the host is quiet. Report hardware, thread count, layout, warmup and repetitions. No phone latency claim.

## Components and execution safety

New files form a separate experiment: `skin_face_transfer_data.py` (role-safe source adapters and deterministic sampler), `skin_face_transfer_study.py` (immutable recipe, budget selection and data bindings), `skin_face_transfer_run.py` (CPU/CUDA preflights, training receipts), `skin_face_transfer_evaluate.py` (post-selection test/appearance evaluation), `skin_face_transfer_report.py` (artifact verification and full comparison).

Data and outputs live at `D:/Luma-RnD/skin_face_transfer_v1`. Register sources and immutable inputs after code tests pass. Actual CPU preflight uses copies of the real Seg1 weights and real TRAIN-only inputs, two updates on batch two for every seed/arm. Its outputs are discarded; it must not initialize CUDA or be used as a native quality/timing claim. CUDA preflight uses the production batch 32 only when the earlier HR and P3 workflows are sealed, actual workers are gone and the GPU is free.

Strict queue order remains HR primary → HR audit/runtime/report → P3 CUDA gate and primary → P3 audit/runtime/report → Seg2 CUDA gate/primary. A missing prerequisite is a refusal before CUDA setup or creating a production job. No second GPU job or CPU latency benchmark while an earlier primary is live. Training receipts/checkpoints are write-once. Retry telemetry replacement on PermissionError and allow only noncritical progress to be skipped; never overwrite or skip source/selection/checkpoint receipts. A failed existing production run requires a separately documented recovery; no silent restart.

## Acceptance

Role guards, target conversion, sampler pairing/prefixes, checkpoint selection and inference serialization have meaningful CPU tests. Real TRAIN preflight demonstrates all six planned paths with the frozen initialization. All future CUDA training, full validation, selection freezing, test inference, actual runtime and verified comparison remain required before a Seg2 improvement claim. Preparation alone does not complete the broader goal.
