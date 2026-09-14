# ChromaSeed-P implementation plan

User-authorized autonomous execution in the existing isolated worktree, without sub-agents. Apply numerical tests before implementation and evidence before completion. No old frozen source edits.

**Goal:** test whether perceptual output weighting and longer iterative readout training improve real instrument-referenced color accuracy at the same compact inference size.

**Architecture:** shared KE128 basis, five readout objectives, person-disjoint selection, independently audited final predictions and full-fit timing. All fitted refinements collapse into the existing predictor.

**Spec:** [registered protocol](../../research/chromaseed_perceptual_v1_protocol.md).

- [x] Add numerical tests and core `scripts/chromaseed_perceptual.py`: local metric, coupled ridge, robust trajectories, fixed predictor payload; confirm RED then GREEN.
- [x] Add person-disjoint bank/selection runner in `scripts/chromaseed_perceptual_train.py`, policy tests, source/parent locks and baseline checks; run all registered banks with progress receipts.
- [x] Implement independent analytic-tensor/SVD audit, recalculate all choices/final scores and inspect losses/negative controls.
- [x] Measure real single-model training and inference in `scripts/chromaseed_perceptual_runtime.py`; keep timings isolated.
- [x] Create measured report/figures, model card, verification receipt and next decision; update active-goal ledger from actual results. Broad goal remains active unless ordinary full-scope requirements are established.
