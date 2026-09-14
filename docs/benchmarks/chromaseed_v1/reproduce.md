# Reproduce ChromaSeed with separate output directories

The authoritative run must remain unchanged. Use NEW directory names for a replay. The original licensed TRAIN cache must have the protocol hash. No test/validation substitutions are permitted.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskReplay = 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search\experiments\runs\chromaseed_v1_replay'
$taskFrozenReplay = 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search\experiments\runs\chromaseed_frozen_v1_replay'
$taskReport = 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search\experiments\runs\chromaseed_v1_replay\report'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed.py tests/test_chromaseed_training.py tests/test_chromaseed_frozen.py -q
& $taskPython scripts/chromaseed_train.py --cache $taskCache --run $taskReplay --stage pretrain --device cuda
& $taskPython scripts/chromaseed_train.py --cache $taskCache --run $taskReplay --stage fit --device cuda
& $taskPython scripts/chromaseed_frozen.py --cache $taskCache --source-run $taskReplay --run $taskFrozenReplay --stage fit
# All choices and final weights now exist; evaluation follows freezing.
& $taskPython scripts/chromaseed_train.py --cache $taskCache --run $taskReplay --stage evaluate --device cuda
& $taskPython scripts/chromaseed_frozen.py --cache $taskCache --source-run $taskReplay --run $taskFrozenReplay --stage evaluate
& $taskPython scripts/chromaseed_audit.py --cache $taskCache --run $taskReplay --frozen-run $taskFrozenReplay --output $taskReport
& $taskPython scripts/chromaseed_report.py --run $taskReplay --frozen-run $taskFrozenReplay --output $taskReport
```

PyTorch deterministic mode and disabled TF32 are part of the recorded run. The code checks source and artifact bindings; modified sources require new run directories. Exact timings depend on process load/device state. The first authoritative frozen-head run overlapped part of GPU fitting; do not interpret its recorded milliseconds as a dedicated hardware benchmark. A replay cannot turn historically exposed cohorts into fresh independent evidence.

The report/audit scripts perform no model selection or training. Report rows average per-model metrics across seeds. Synthetic diagnostic losses and skin instrument errors are separate endpoints. Additional feature-gap diagnosis uses only the mixed fit18 image features and never modifies this version's renderer.
