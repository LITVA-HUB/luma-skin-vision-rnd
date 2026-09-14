# Reproduce ChromaSeed-G

Use the existing verified P banks, D source/verification and original TRAIN cache. Preserve all existing runs; choose a new replay directory. No dependencies or downloads are required. Recorded Python3.12.6, NumPy2.5.3, SciPy1.18.1, one CPU thread. Torch is used by inherited training infrastructure, while the standalone predictor imports NumPy only.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskRun = 'experiments/runs/chromaseed_gated_v1_replay'
$taskOutput = 'experiments/runs/chromaseed_gated_v1_replay/report'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed_gated.py tests/test_chromaseed_perceptual.py tests/test_chromaseed_perceptual_protocol.py tests/test_chromaseed_condensed_exact.py -q
& $taskPython -m pytest tests/test_chromaseed_gated_numpy.py -q
& $taskPython scripts/chromaseed_gated_train.py run --cache $taskCache --run $taskRun
& $taskPython scripts/chromaseed_gated_audit.py --cache $taskCache --run $taskRun --output $taskOutput
& $taskPython scripts/chromaseed_gated_runtime.py --cache $taskCache --run $taskRun --output $taskOutput
& $taskPython scripts/chromaseed_gated_report.py --run $taskRun --output $taskOutput
```

Check each exit before continuing; run timing sequentially after all heavy jobs finish. Report Markdown links are authored for the canonical docs location and need relocation if publishing a replay report elsewhere. The replay report writer emits measured tables but canonical narrative timings describe the original run; use replay runtime.json/summary.json for new machine timing.

Original execution:23 primary tests passed in1.56s before fitting;3 separate portable tests passed in0.25s. Primary PID36812 completed all nine inner banks, froze24 choices, then completed all three final banks before evaluating72 selected models. No bank resume occurred. Primary workflow6.671s excludes interpreter imports/audit/runtime. The2,016 readouts include864 exact fallback copies;360 positive residual coefficient solutions share120 algebra operations, with36 perceptual base solves and4 gate fits. Do not describe every stored configuration as an independent training run.

Independent audit session27067 returned exit0 and reconstructs all72 selected fits plus12 extra positive soft/hard probes using separate gate least squares, direct kernels and analytic-tensor/SVD readouts. Every stored OOF output, selected output and exact P/fallback payload is checked. Maximum independent native-Lab drift2.46e-5 is below the registered0.001 tolerance. The audit shares the previously verified CIEDE2000 implementation and frozen roles.

Runtime returned exit0 and checks canonical and standalone batch-one calls for all72 selected models;96 selected full fits plus16 fixed active fits include warmups and must reproduce every stored array exactly. The same-run static NumPy control is retained. Construction/validation timing excludes archive I/O; inference excludes preprocessing, imports and temporary-process memory. No GPU inference claim is made.

Only TRAIN color/target/patient/site/device arrays and locally recorded model artifacts are used. No raw images/tokens, direct identifiers, legacy validation/calibration/test or data acquisition. Existing split reuse limits inference about future users and phones.

[Protocol](../../research/chromaseed_gated_v1_protocol.md) · [Report](report.md) · [Verification](verification.json) · [Model card](../../architecture/chromaseed_gated_model_card.md).
