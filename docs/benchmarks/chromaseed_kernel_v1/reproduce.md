# Reproduce ChromaSeed-K v1

Use new directories for replay. Preserve the original K, R, palette and local-search sources/artifacts. All data roles are already exposed exploratory roles; reproducing them is not new confirmation.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskRun = 'experiments/runs/chromaseed_kernel_v1_reproduction'
$taskReplay = 'experiments/runs/chromaseed_kernel_v1_reproduction_timing'
$taskLegacy = 'experiments/runs/skin_local_search_v1'
$taskOutput = 'experiments/runs/chromaseed_kernel_v1_reproduction/report'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed_kernel.py tests/test_chromaseed_kernel_bank.py -q
& $taskPython scripts/chromaseed_kernel_train.py run --cache $taskCache --run $taskRun --legacy-run $taskLegacy
& $taskPython scripts/chromaseed_kernel_audit.py --cache $taskCache --run $taskRun --legacy-run $taskLegacy --output $taskOutput
& $taskPython scripts/chromaseed_kernel_runtime.py --cache $taskCache --run $taskRun --legacy-run $taskLegacy --output $taskOutput
& $taskPython scripts/chromaseed_kernel_replay_timing.py --cache $taskCache --original-run $taskRun --replay-run $taskReplay --legacy-run $taskLegacy --output $taskOutput
& $taskPython scripts/chromaseed_kernel_report.py --run $taskRun --output $taskOutput
```

Check each command's exit code before continuing; a failed audit is not a passing run. Full training stages are sequential: all nine inner banks → frozen primary/exit/blend choices → all three final banks → outer metrics. Individual `inner`, `select`, `final`, `evaluate` stages are also supported. Before resuming an interrupted run, verify its process is terminal; completed banks are hash checked and reused. No mutation of frozen sources is allowed as a way of making a previous run appear reproducible.

The original measured K run contains3321 inner and1107 final analytical readouts in12 banks, with123 primary +9 adaptive +18 blend models evaluated and3 width-one reference reproductions. This uses common kernels and spectral algebra, not4428 separate full preprocessing/training jobs. The runtime script additionally performs88 standalone fits including warmups. The complete replay repeats the same4428 configurations once, with process-level time and array comparisons; do not count this as a new search or independent evidence.

CPU/NumPy2.5.3, PyTorch2.8.0+cu128, Python3.12.6. One thread, deterministic torch, TF32 off. RTX4060 handles full spectral solves; compact algebra uses CPU after a synthetic preflight. Legacy selected guided-RBF/KRR artifacts and hashes are required; their predictor remains unchanged. No network or pretrained weight download is needed.

Only color36, native Lab and patient/site/device metadata are loaded from original TRAIN; no image arrays or legacy validation/calibration/test. Raw selected weights and row-level predictions remain in ignored local runs. Aggregate reports do not contain participant identifiers or images. The auditor uses its own direct kernel/SVD refit and the previously audited metric helper; [verification.json](verification.json) binds its dependencies in addition to the primary source lock.

The documentation/model-loading example uses a fixed seed and does not select the best seed from outer errors. The report requires the passing audit, matching runtime/audit hash, a complete replay receipt and unchanged source/selection bindings.
