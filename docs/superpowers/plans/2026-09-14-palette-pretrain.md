# Palette encoder pretraining implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. No subagent delegation is authorized.

**Goal:** Produce verified, matched palette-trained and shuffled-target encoder initializations compatible with three existing large ChromaSeed architectures.

**Architecture:** A TRAIN-only preparation stage creates real spatial spectral patches and fixed simulated photometric views. Six small auxiliary networks train the existing shared local encoder, whose parameters are then transferable under native fit-only normalization. The running HR process remains separate.

**Tech stack:** Existing Python, NumPy, SciPy, Pillow and PyTorch environment; CPU FP32, one thread.

**Spec:** docs/research/chromaseed_palette_pretrain_v1_protocol.md

## Constraints

- Preserve all original HR/P1 and earlier frozen sources.
- Original 19 UMINHO TRAIN sources only; no held cubes/native targets.
- No additional GPU job or latency benchmark during HR.
- Fixed 128 valid spatial patches per source and 8 views per patch.
- No quality claim from auxiliary reconstruction/training loss.

## Tasks

- [x] Add feature-order, patch-boundary, paired-fitter and normalization-transfer tests; observe missing-module failures. Target grouping is additionally checked on every prepared example by the independent data auditor.
- [x] Implement new data preparation, shared encoder and CPU fitter modules without editing frozen modules.
- [x] Verify synthetic behavior and exact compatible initialization, then freeze sources and input bindings.
- [x] Prepare and read back the spatial patch dataset from its original measured sources.
- [x] Run six fixed2048-step auxiliary fits and verify exported NumPy outputs and normalization transfer.
- [x] Record results and the remaining matched native-transfer experiment; retain the complete HR verification work as pending until its actual primary completion.

Outcome: [P2 report](../../benchmarks/chromaseed_palette_pretrain_v1/report.md). All auxiliary runs and audits completed successfully; four tests passed. Source lock db5e5c8d9eef8f7a066a0fac8296956507ceb59522b8ffb95b7f86ab8b5b2cf7. Native fine-tuning is a separate, uncompleted experiment, and the whole goal remains active.
