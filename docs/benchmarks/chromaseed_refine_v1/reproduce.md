# Reproduce ChromaSeed-R v1

Use a new run directory for a complete replay. The original run and its frozen sources must remain unchanged. A replay of these exposed exploratory roles is not fresh confirmation.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskRun = 'experiments/runs/chromaseed_refine_v1_replay'
$taskOutput = 'experiments/runs/chromaseed_refine_v1_replay/report'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed_refine.py tests/test_chromaseed_refine_training.py tests/test_chromaseed_refine_numpy.py -q
& $taskPython scripts/chromaseed_refine_train.py run --cache $taskCache --run $taskRun --device cuda
# The run command finishes all inner fits and freezes selections before final refits,
# then finishes all refits before evaluating any outer labels.
& $taskPython scripts/chromaseed_refine_audit.py all --cache $taskCache --run $taskRun --output $taskOutput
& $taskPython scripts/chromaseed_refine_report.py --run $taskRun --output $taskOutput

# Optional registered storage follow-up: no extra training or threshold tuning.
$taskPrecisionRun = 'experiments/runs/chromaseed_refine_precision_v1_replay'
& $taskPython scripts/chromaseed_refine_precision.py --source-run $taskRun --cache $taskCache --run $taskPrecisionRun --output $taskOutput
& $taskPython scripts/chromaseed_refine_precision_diagnosis.py --source-run $taskRun --precision-run $taskPrecisionRun --cache $taskCache --output $taskOutput
& $taskPython scripts/chromaseed_refine_report.py --run $taskRun --output $taskOutput
```

The training command may be resumed after a process interruption; first verify no matching process is still running. Completed banks are checked against their source lock and artifact hashes. A partially completed bank is rerun deterministically; completed banks are not retrained. Optional individual stages are `inner`, `select`, `final`, `evaluate`; all use the same arguments and source lock. CUDA graph warmup is reset to initial parameters, empty Adam moments and step zero. Recorded setup cost includes capture and sampling. The exact hardware/run time is not portable.

Only `color`, `tokens`, `target`, `patient`, `site`, and `device` are loaded from the original TRAIN cache. Legacy validation/calibration/test and image arrays are excluded. Raw weights and row-level predictions remain in the ignored local run directory. Report files contain aggregate evidence. The independent NumPy path and audit have separate hashes in `audit.json` because they were implemented while the immutable training run executed.

Full research configuration and provenance: [prospective protocol](../../research/chromaseed_refine_v1_protocol.md). No external model weights or paper implementation were downloaded for this series. Existing dataset usage/license restrictions remain applicable; a completed experiment does not grant production or redistribution rights.
