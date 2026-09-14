# Reproduce ChromaSeed-D

Use the existing verified S manifest/verification and a fresh D run directory. No downloads or dependency changes. Recorded Python3.12.6, NumPy2.5.3 and SciPy1.18.1, one CPU thread; scikit-learn supplies independent AUC/balanced-accuracy checks in the audit.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskRun = 'experiments/runs/chromaseed_camera_support_v1_replay'
$taskOutput = 'experiments/runs/chromaseed_camera_support_v1_replay/report'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed_camera_support.py -q
& $taskPython scripts/chromaseed_camera_support_run.py --cache $taskCache --run $taskRun
& $taskPython scripts/chromaseed_camera_support_audit.py --cache $taskCache --run $taskRun --output $taskOutput
& $taskPython scripts/chromaseed_camera_support_report.py --run $taskRun --output $taskOutput
```

Check each exit before continuing. Eleven tests passed in0.38s after formatting and before the real run. Original primary PID37356 returned exit0 directly. The first audit passed its numerical assertions but returned exit1 while serializing a NumPy int64 counter. Only the audit writer was corrected to emit Python scalars; the entire final audit then returned exit0 directly. No primary fit was changed or repeated. Final report generation also returned exit0 directly.

The audit independently reconstructs all24 classifier pipelines using SVD, all7728 query scores, pooled/fold metrics, all15 caliper problems by exhaustive enumeration, all40 matched metric groups and six support calculations. It shares the previously verified CIEDE2000 formula and frozen folds. Stored arrays use ordinal indices only and stay local; no patient identifiers, images or tokens.

Classifiers train on acquisition labels, not skin-color targets. The native-Lab views are oracle diagnostic controls. Color coverage radii are descriptive, not a deployment rejection policy. No prior outer results or legacy validation/calibration/test are read.

[Protocol](../../research/chromaseed_camera_support_v1_protocol.md) · [Report](report.md) · [Verification](verification.json).
