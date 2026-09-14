# Reproduce ChromaSeed-X

Existing original TRAIN, A/G banks and A verification are required. Keep original runs intact and use a new replay directory. No dependency installation or downloads are needed. Recorded Python3.12.6, NumPy2.5.3, SciPy1.18.1; analytic training is CPU one thread. Some inherited training utilities import Torch; the standalone predictor imports only NumPy.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskRun = 'experiments/runs/chromaseed_projection_v1_replay'
$taskOutput = 'experiments/runs/chromaseed_projection_v1_replay/report'
if (Test-Path -LiteralPath $taskRun) { throw 'Choose a new replay directory.' }
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $taskPython -m pytest tests/test_chromaseed_projection.py tests/test_chromaseed_projection_reference.py -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $taskPython scripts/chromaseed_projection_train.py run --run $taskRun --cache $taskCache
if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
& $taskPython scripts/chromaseed_projection_audit.py --run $taskRun --cache $taskCache --output $taskOutput
if ($LASTEXITCODE -ne 0) { throw 'Independent audit failed.' }
# Run after other heavy processes finish:
& $taskPython scripts/chromaseed_projection_runtime.py --run $taskRun --cache $taskCache --output $taskOutput
if ($LASTEXITCODE -ne 0) { throw 'Runtime failed.' }
& $taskPython scripts/chromaseed_projection_report.py --run $taskRun --output $taskOutput
if ($LASTEXITCODE -ne 0) { throw 'Report failed.' }
```

Report narrative describes the canonical experiment; replay measurements are in its runtime/summary files. Markdown links target the canonical docs location and need relocation if sharing a replay report elsewhere. Do not run old G/GS verifier mains because they rewrite now-bound receipts. A and X verifiers check their existing canonical receipts read-only after creation; preserve their SHA256 values.

Original execution:12 primary tests passed1.66s after formatting and before freezing59 source/145 input bindings. PrimaryPID36304/session37948 exited0 after nine inner banks,24 frozen choices, three final banks and111 evaluations. No bank reuse. Workflow20.1129713s excludes imports/audit/runtime.5,772 stored configurations are not5,772 independent fits:3,744 solutions,1,872 aliases,144 imported controls,12 constants;624 Gram decompositions/468 bases/156 widths/12 covariances/4 gates and36+36 auxiliary solves.

Reference tests3 passed1.52s before audit. Audit72869 exited0 in49.007s: weighted-design SVD, induced projection distance operators,156 exhaustive widths,468 dense pivot paths, all OOF/final predictions and policies,72 independent final refits+12 positive probes. Component eigenvectors inside tied eigenspaces are coordinate conventions; their induced metric is the invariant check. Max operator-relative difference3.17e-8, eigenvalue difference4.27e-14, reference prediction difference2.29e-5Lab. Stored widths agree exactly. Actual standalone predictions differ by at most1.95e-12Lab.

Runtime exited0 after a fresh scan found no other Python worker. All111 consumers use20 warmups and3 query passes;24 selected seed17 settings×4 full fits plus4 fixed positive settings×4 total112, all payload arrays exact. Report followed profiling. Final combined suite15 passed1.72s. A postprocessing import-order lint issue and a PowerShell display-pipeline typo were corrected before final checks; frozen primary sources and model results were not changed.

Only original TRAIN color/target/patient/site/device and existing local artifacts were used. No raw images, query targets during fitting, legacy validation/calibration/test, data acquisition, publication or delegation. The same historical people/roles limit generalization claims.

[Protocol](../../research/chromaseed_projection_v1_protocol.md) · [Report](report.md) · [Verification](verification.json) · [Model card](../../architecture/chromaseed_projection_model_card.md).
