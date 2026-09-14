# Reproduce ChromaSeed-P

Use fresh output directories, the original TRAIN cache and existing KF/KE parent runs. No network or new package installation is required. Recorded environment: Python3.12.6, NumPy2.5.3, SciPy1.18.1, PyTorch2.8.0+cu128. Preserve original source locks and results. Check every command exit before proceeding.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskRun = 'experiments/runs/chromaseed_perceptual_v1_replay'
$taskOutput = 'experiments/runs/chromaseed_perceptual_v1_replay/report'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed_perceptual.py tests/test_chromaseed_perceptual_protocol.py tests/test_chromaseed_condensed_exact.py -q
& $taskPython scripts/chromaseed_perceptual_train.py run --cache $taskCache --run $taskRun --parent-run experiments/runs/chromaseed_fast_kernel_v1 --exact-run experiments/runs/chromaseed_condensed_exact_v1
& $taskPython scripts/chromaseed_perceptual_audit.py --cache $taskCache --run $taskRun --parent experiments/runs/chromaseed_fast_kernel_v1 --output $taskOutput
& $taskPython scripts/chromaseed_perceptual_runtime.py --cache $taskCache --run $taskRun --output $taskOutput
& $taskPython scripts/chromaseed_perceptual_report.py --run $taskRun --output $taskOutput
```

All9 inner banks finish before15 choices freeze; all3 final banks finish before outer scores.243 saved configurations/bank and2916 total reuse shared algebra. Checkpoint0 iterative candidates alias existing normalized-ridge models and are not extra fits. Optional train stages: `inner`, `select`, `final`, `evaluate`. Source or receipt mismatch raises; do not alter a frozen run to force resumption.

The independent audit checks source/bank hashes, normalizers,413 100 inner prediction rows, all648 monotone trajectory logs,15 choices and45 selected refits/17 970 prediction rows. It uses analytic color tensors and an explicitly augmented least-squares/SVD solve. Six fixed inner-fold0/seed17/width1/alpha0.1 trajectories yield18 positive-step probes at1/4/16, ensuring audit coverage even when all selected iterative models use0. This is targeted independent numerical replay, not a full independent re-fit of every one of2916 configurations.

Runtime performs45 actual batch-one inference checks,60 selected full-fit runs including warmups, and24 extra fixed16-step timing runs including warmups. The extra cases use the same seed17/width1/alpha0.1 settings as the analytic probes, on each final fit set. They add cost evidence only; no extra outer predictions or quality selection. Profile without competing training or heavy tasks. Inference excludes image processing; training timings exclude I/O and hyperparameter search.

Original run training37639/PID39176 and audit50020 are terminal exit0; runtime returned exit0 directly. An old progress PID does not establish a live job. [Verification](verification.json) · [Protocol](../../research/chromaseed_perceptual_v1_protocol.md) · [Report](report.md).
