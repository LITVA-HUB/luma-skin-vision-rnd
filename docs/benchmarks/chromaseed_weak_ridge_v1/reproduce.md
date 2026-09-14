# Reproduce ChromaSeed-W

Use fresh run/report directories, the existing verified P parent and KE source lock, and original TRAIN only. Recorded environment: Python3.12.6, NumPy2.5.3, SciPy1.18.1, PyTorch2.8.0+cu128. No downloads or package changes are needed. Preserve all previous results and check every command exit before continuing.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskRun = 'experiments/runs/chromaseed_weak_ridge_v1_replay'
$taskOutput = 'experiments/runs/chromaseed_weak_ridge_v1_replay/report'
$taskParent = 'experiments/runs/chromaseed_perceptual_v1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed_weak_ridge.py tests/test_chromaseed_perceptual.py tests/test_chromaseed_perceptual_protocol.py tests/test_chromaseed_condensed_exact.py -q
& $taskPython scripts/chromaseed_weak_ridge_train.py run --cache $taskCache --run $taskRun --parent-run $taskParent --exact-run experiments/runs/chromaseed_condensed_exact_v1
& $taskPython scripts/chromaseed_weak_ridge_audit.py --cache $taskCache --run $taskRun --parent $taskParent --output $taskOutput
& $taskPython scripts/chromaseed_weak_ridge_runtime.py --cache $taskCache --run $taskRun --output $taskOutput
& $taskPython scripts/chromaseed_weak_ridge_report.py --run $taskRun --parent $taskParent --output $taskOutput
```

All9 inner banks finish before15 choices freeze; all3 final banks finish before scoring45 selected models. Each bank saves729 configurations;8748 total include2916 exact-P controls. Shared basis algebra and saved step checkpoints do not count as independent full trainings. Alpha0.1/1/10 controls are checked with exact payload equality after index remapping, including old positive correction checkpoints. No old P source is modified or monkey-patched at runtime.

The final audit independently checks all stored OOF predictions and matching P arrays, all15 choices and45 selected final analytic-tensor/SVD refits. Its fixed positive-step probes use inner fold0/seed17/width1/alpha0.0001 for both iterative families in each role, at1/4/16 steps. Every other saved model is covered by source/bank/prediction and trajectory checks; this is not a full independent retraining of8748 models.

Profile after the audit exits, without competing heavy jobs.45 actual batch-one predictors,60 selected full fits including warmups,24 extra fixed16-step cost fits at seed17/width1/alpha0.0001. Extra cost fits do not generate new outer quality scores. Fixed-step cases can execute fewer than16 solves after a rejected direction; requested checkpoints and executed work are reported separately.

Original completed handles: training46551/PID33824, initial audit60251, final augmented audit76553, all exit0; runtime returned exit0 directly. Inspect authoritative process/session state before resuming anything; an old progress PID is not sufficient. [Verification](verification.json) · [Protocol](../../research/chromaseed_weak_ridge_v1_protocol.md) · [Report](report.md).
