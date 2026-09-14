# Reproduce ChromaSeed-GS

Use frozen G selected artifacts and verification. Choose a new run directory: completed diagnostic results cannot be overwritten. No fitting, downloads or dependency updates are involved. Recorded environment: Python3.12.6, NumPy2.5.3, SciPy1.18.1, one CPU thread.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskRun = 'experiments/runs/chromaseed_gate_stability_v1_replay'
$taskOutput = 'experiments/runs/chromaseed_gate_stability_v1_replay/report'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed_gate_stability.py -q
& $taskPython scripts/chromaseed_gate_stability_run.py --cache $taskCache --run $taskRun
& $taskPython scripts/chromaseed_gate_stability_audit.py --cache $taskCache --run $taskRun --output $taskOutput
& $taskPython scripts/chromaseed_gate_stability_report.py --run $taskRun --output $taskOutput
```

Check each command's exit before continuing. Report links target the canonical docs layout and need relocation for a replay elsewhere. The report preserves canonical process/test evidence; current diagnostic/audit wall times come from the supplied run. Diagnostic time is not a production runtime measurement.

Original tests first failed because the new core module did not yet exist. Eight tests then passed; imports were formatted before a final8-pass run in1.51s and source freeze. Original primary PID39468 returned exit0 directly, then audit session1166 returned exit0 as confirmed through its live session handle. The primary core, runner, tests and protocol were not changed after execution began.

The independent auditor reconstructs every transformed feature vector with separate channel-wise algebra and all948,816 query predictions with the NumPy-only batch-one consumer. It recomputes every transform/dose person/camera summary, every gate sign, all534 unique boundary input pairs repeated across six models, every matched-control pair output and all1,392 nearest-feature projections/limit jumps. Existing CIEDE2000 and frozen row indexing remain shared dependencies. Maximum ordinary/boundary native-Lab drift is2.92e-12/1.80e-12.

Only TRAIN arrays and frozen G model/prediction artifacts are loaded. Boundary pairs need not be near unmodified inputs: median initial mixture is15.69%, despite pair separation≤0.0002 RGB. The unconstrained feature projection is not a guaranteed realizable image. No new camera/target measurements, real-phone benchmark, changed weights or inference speedup is claimed.

[Protocol](../../research/chromaseed_gate_stability_v1_protocol.md) · [Report](report.md) · [Verification](verification.json).
