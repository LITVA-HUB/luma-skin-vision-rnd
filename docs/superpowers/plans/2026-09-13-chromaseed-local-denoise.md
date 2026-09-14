# ChromaSeed local denoising implementation plan

> For agentic workers: execute inline with superpowers:executing-plans. No delegation under the current worktree instructions. User has already authorized autonomous bounded experiments.

**Goal:** Measure whether block-local denoising improves a roughly 11 KB skin-color regressor's quality/training/response tradeoff.

**Architecture:** Five matched small networks, independent optimizer slots per block, target-only noisy training states and deterministic target-free deployment. Reuse fixed roles and analytical references; isolate all new code and evidence.

**Tech stack:** Existing NumPy, PyTorch/CUDA, pytest; no new dependencies.

**Spec:** [ND protocol](../../research/chromaseed_local_denoise_v1_protocol.md).

**Constraints:** Original TRAIN only, three fixed person folds, seeds17/29/43, rates.001/.003, checkpoints512/2048/8192, batch64. FP32 deterministic RTX4060 training and single-thread CPU prediction. Preserve all old frozen sources and results. New candidates do not change NS.

- [x] Inventory R, G/H and previous frozen heads; write exact architecture, noise, loss, selection and accounting rules.
- [x] Write behavioral tests first in tests/test_chromaseed_local_denoise.py; run to observe missing implementation, then implement scripts/chromaseed_local_denoise.py and scripts/chromaseed_local_denoise_numpy.py. Test local gradients, stage semantics and independent optimizer slots, not document text.
- [x] Implement scripts/chromaseed_local_denoise_fit.py: fit(x,y,weights,family,slots,steps,checkpoints,device,engine) returns checkpoint payloads and complete timing trace. Verify CPU bank versus singleton and CUDA eager versus fully reset graph on synthetic data.
- [x] Implement scripts/chromaseed_local_denoise_train.py with immutable source/data/parent locks, completed-bank receipts, nested selection and saved final stage/transform predictions. Preflight counts and smoke before any real fit.
- [x] Run 45 inner banks, freeze 15 choices, then 45 final singleton fits and import 18 exact analytical references. Keep actual live process handle; continue it across observation timeouts.
- [x] Implement/run independent scripts/chromaseed_local_denoise_audit.py: all normalizers, inner stage outputs,90 scores/15 choices, final outputs/metrics, actual consumer calls and paired descriptive intervals. No changes to frozen primary code after fitting.
- [x] Run scripts/chromaseed_local_denoise_runtime.py: actual CPU query timing and45 complete CUDA replay fits, including construction and export.
- [x] Generate report/CSV/model card/next decision with scripts/chromaseed_local_denoise_report.py; seal hashes, run relevant tests and repeat verifier in read-only mode. Check scoped diff; update mutable status/goal with actual outcome.

Expected check examples: an oracle clean predictor reaches its correct final endpoint under the registered recurrence; state-only second block receives sqrt(.5) times the first constant prediction for K2; a local block's loss has zero derivative with respect to other blocks, while connected e2e gradients can cross blocks. A corrupted final output must fail independent prediction/metric reconstruction. Invalid input shape or nonfinite color must be rejected by the actual consumer.

Execution evidence, 2026-09-13: initial11 tests failed on absent ND implementation. After implementation, a missing scripts import path in the test harness was diagnosed against existing tests and corrected. Final18 tests pass, including CPU bank independence, local versus connected gradients, noise independence, optimizer reference and GPU eager/graph parity. Preflight passed before source freeze.

Primary70999/PID35492 failed on Windows atomic progress-file replacement after33 completed banks. Authoritative session exit1 and absent PID established termination. The same replacement later succeeded; suspected concurrent PowerShell read contention is documented, not asserted proven. Resume28365/PID16888 preserved all frozen sources and reused33 banks, then completed exit0. Audit31986 and runtime8779 completed exit0.90 completed banks/810 inner checkpoint models/45 final models/18 exact controls; all45 timing refits bitwise equal. Full audit and18 tests pass.

Sealed verification36442da6dabe927b4056ef5532fc7a6dd3956c14eb8ed35736a3f161ac3e2376; subsequent report call passed read-only with unchanged hash. Additional in-memory constant-oracle endpoint and corrupted-prediction rejection checks passed. No worker remains. Mutable status/goal updated; broader goal active. Prefix study is the next planned question and has not started.
