# Seg2 face transfer implementation plan

> Execute inline in the existing isolated worktree using `superpowers:executing-plans`. AGENTS.md prohibits delegation. Autonomous experiment authorization is already present; no repeated approval checkpoint is required.

**Goal:** Test whether cleaned extra facial masks and more fine-tuning steps improve Seg1 under matched update budgets.

**Architecture:** Preserve the frozen width-24 SkinUNet and initial checkpoint. Add role-safe dataset adapters, paired sampling, a six-run experiment and complete post-selection comparison as separate sources.

**Tech Stack:** Existing Python 3.12 R&D environment, NumPy, Pillow, PyTorch, pytest. No environment/lock changes.

**Spec:** `docs/superpowers/specs/2026-09-14-face-transfer-design.md`.

## Global constraints

No edits to frozen sources or datasets. No subagents. No CUDA job or CPU latency measurement while HR/P3 is live. Windows JSON is UTF-8. Inputs and receipts use exact hashes and real process state. CelebA indices are global row indices into the full arrays; never replace them with `range(len(indices))`.

## Task 1: role-safe batches and paired sample streams

Files: new `scripts/skin_face_transfer_data.py`, `tests/test_skin_face_transfer_data.py`.

Interfaces: `ArraySplit(source, role, rgb, labels, indices, encoding)` exposes `get(global_ids)` returning uint8 RGB and binary masks; `PairedSampler(lapa_ids, celeba_ids, seed, batch_size).next(arm)` returns source IDs and global row IDs. Separate sampler instances per arm must share anchors and preserve shorter-run prefixes. `load_split(source, role, allow_test=False)` refuses test access by default.

- [x] Write tests: gapped allowed IDs exclude held rows; unknown/negative IDs fail; LaPa labels 1/6 and binary CelebA map correctly; invalid labels/shapes fail; paired anchor IDs match for 20 consecutive steps and prefixes survive reconstruction; additional samples come only from the configured TRAIN indices.
- [x] Run `python -m pytest tests/test_skin_face_transfer_data.py -q`; verify failure from missing functionality.
- [x] Implement adapters and sampler, then run the tests and targeted ruff. Actual TRAIN and VALIDATION adapters also read successfully at their registered counts.

## Task 2: fixed budgets, choices and data registration

Files: new `scripts/skin_face_transfer_study.py`, `tests/test_skin_face_transfer_study.py`.

Interfaces: `recipe()` returns the six trajectories and fixed budgets; `select_prefix(history,budget)` ranks source-balanced validation scores at or before the requested budget, preserving step zero and earliest ties. `freeze_registration()` binds source files, source array/index/metadata receipts and the original Seg1 checkpoint before any held model evaluation.

- [x] Test that better checkpoints after a budget cannot win its prefix; a worse fine-tune loses to step zero; ties are stable; NaN and missing-source metrics fail. Test the exact six-run/equal-update/sample-count recipe.
- [x] Implement selection and registration, execute tests and targeted ruff. Registration c92d12e53e42325cec76fd6ad61dc94a8c685fcba9a48b97ccc0a791d108da88 binds 63 source/input files. These bound files are now immutable.

## Task 3: actual CPU preflight and prospective GPU runner

Files: new `scripts/skin_face_transfer_run.py`, `tests/test_skin_face_transfer_run.py`.

Interfaces: CLI `preflight-cpu`, `preflight-cuda`, `run`. Use fresh copies of `SkinUNet` initialized with the verified Seg1 state. CPU preflight performs two batch-two steps on real TRAIN inputs for each arm/seed, with finite loss/gradients and exact initial-state equality. Production validates every 498 updates through 5,976 and preserves each checkpoint and receipt.

- [x] Test own checkpoint round-trip with actual tensors; corrupted/missing initialization fails; prior-live/absent-seal guard refuses before GPU setup; progress replacement is optional but immutable receipts cannot overwrite.
- [x] Implement the runner, run CPU tests, then freeze final registration and execute the actual six-case CPU gate. Final 32 tests passed in 2.39 s; targeted ruff clean. Registered CPU gate session56744 terminal0: six cases, 12 actual TRAIN updates, exact common anchors/augmentations and initialization, no CUDA context. See preflights/cpu_0e21e5ef03bb4cedb14235ab45b436c4/result.json. Earlier development CPU probe is separately preserved and is not the production gate.
- [x] Exercise the actual CUDA/run prerequisite refusal while HR lives, confirm no CUDA context or production artifacts are created and HR is still live. All six CLI paths and six in-process paths refuse before work; guard probe session40421 terminal0.

## Task 4: complete evaluation and comparison consumers

Files: new `scripts/skin_face_transfer_evaluate.py`, `scripts/skin_face_transfer_report.py`, companion tests.

- [x] Implement and freeze post-selection consumers. Tests cover exact mask/confusion arithmetic, color coverage, stored-mask corruption, late selection changes, complete paired sample traces and all 18 choices/9 data contrasts/12 duration contrasts, including negative outcomes. Real full evaluation and timings remain pending below.
- [ ] Freeze all 18 prefix choices from complete validation histories before loading TEST. Evaluate each unique selected checkpoint and unchanged Seg1 on both sources; save per-image confusion and apparent-color metrics with source/checkpoint hashes.
- [ ] Verify complete source/seed/arm/budget coverage, arithmetic, missing-mask coverage and all contrasts. Tests must reject incomplete runs, changed checkpoint data and a late test-based choice.
- [ ] Record actual model size/parameter count and quiet-host CPU timings. Report every arm and budget, including regressions; explicitly distinguish segmentation and apparent-image color from native instrument color.

## Task 5: production execution after prerequisite queue

- [ ] Observe actual HR terminal success, execute its frozen full verification, then P3 CUDA/primary/full verification before Seg2 CUDA.
- [ ] Run Seg2 production batch preflight, all six trajectories, freeze all choices, evaluate and verify the full comparison. A final success claim requires actual artifacts and results; preparing consumers is not model improvement.

Plan review: the data-only intervention, extra steps, existing learned initialization, all negative outcomes and the GPU ordering are covered. Run/evaluation phases remain visibly incomplete until executed; no synthetic quality or timing substitutes.
