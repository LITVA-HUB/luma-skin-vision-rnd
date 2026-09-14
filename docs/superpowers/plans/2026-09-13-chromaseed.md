# Luma ChromaSeed Implementation Plan

**Goal:** Test the user's palette-first transfer hypothesis on a compact skin-color regressor and name the research model.

**Architecture:** Deterministic synthetic color-surface generator and tested color36 extraction; shared compact MLP pretraining; nested TRAIN-only transfer experiment with negative and compute controls; immutable selection and separate evaluation.

**Tech Stack:** Existing Python/NumPy/PyTorch CUDA environment; RTX4060; pytest; matplotlib for scientific artifacts.

**Spec:** docs/research/chromaseed_v1_protocol.md.

## Constraints

Use original TRAIN hash only; preserve all previous frozen code/results; no external data download or upload; label synthetic and reused exploratory evidence. Execute locally without delegation. User explicitly authorized autonomous research choices.

- [x] Tests first: color feature parity, color-space target independence, rendering determinism/range, palette provenance and checkpoint serialization.
- [x] Implement generator/core in scripts/chromaseed.py.
- [x] Implement staged runner in scripts/chromaseed_train.py, freeze sources and data, verify plumbing.
- [x] Run palette generation/pretraining, nested skin adaptation, freeze all selections, then evaluate saved models.
- [x] Independently recompute metrics, record full costs and compare controls; save report, graph, model card, current status and reproduction commands.
- [x] Secondary extension: freeze hidden color bases and fit analytic output weights, with random/shuffled controls; separate pre-evaluation protocol and source lock.

Completed:855 authoritative fits/refits,69 relevant tests,135+36 independent output/basis audits. No universal pretraining win. Frozen source files/results preserved. Further renderer changes require a new version; texture-gap diagnostic did not change this version.
