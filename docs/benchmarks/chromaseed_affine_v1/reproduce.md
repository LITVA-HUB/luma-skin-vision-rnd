# Reproduce ChromaSeed-A

Use the existing G banks and frozen GS verification plus original TRAIN only. No dependency install, new data or GPU is needed for this analytic experiment. Keep all original artifacts; use a new replay directory. This run used Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1 and one CPU thread. Inherited training infrastructure imports Torch; the actual portable model consumer imports only NumPy.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskRun = 'experiments/runs/chromaseed_affine_v1_replay'
$taskOutput = 'experiments/runs/chromaseed_affine_v1_replay/report'
if (Test-Path -LiteralPath $taskRun) { throw 'Choose a new replay directory first.' }
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed_affine.py tests/test_chromaseed_affine_reference.py tests/test_chromaseed_gate_stability.py -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $taskPython scripts/chromaseed_affine_train.py run --run $taskRun --cache $taskCache
if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
& $taskPython scripts/chromaseed_affine_audit.py --run $taskRun --cache $taskCache --output $taskOutput
if ($LASTEXITCODE -ne 0) { throw 'Independent audit failed.' }
# Run after all other heavy work is finished:
& $taskPython scripts/chromaseed_affine_runtime.py --run $taskRun --cache $taskCache --output $taskOutput
if ($LASTEXITCODE -ne 0) { throw 'Runtime checks failed.' }
& $taskPython scripts/chromaseed_affine_report.py --run $taskRun --output $taskOutput
if ($LASTEXITCODE -ne 0) { throw 'Report failed.' }
```

The report's narrative describes this canonical study, while replay timings and tables derive from the replay. Canonical Markdown links need relocation if moving a replay report elsewhere. Original control banks and source locks must remain unchanged. Do not rerun old G/GS verifier main functions: they rewrite receipts now bound by A. The A verifier checks the canonical archive; once its receipt exists it verifies without rewriting that receipt.

Primary run: PID35120/session41445, nine inner banks, all 24 policy choices frozen, then three final banks, then evaluations. No bank resume occurred. Workflow 15.9301382 s excludes interpreter imports/audit/runtime. There are 1,884 stored configurations, not 1,884 independent fits: 1,152 new coefficient solutions, 192 Gram eigendecompositions, 36 basis preparations, 4 gate fits, 36+36 auxiliary solves, 576 exact aliases, 144 imported controls and 12 constant fits.

Independent audit session18105 exited0 in 34.944 s. It checks all OOF/final predictions and choices, and independently refits 72 selected new instances plus 12 fixed positive probes using original-fit moments, pinned audited landmarks, direct kernels, an independent gate fit, analytic color metric and augmented-design QR/SVD. It shares verified CIEDE2000 and the frozen person split functions. Maximum independent drift 9.63e-6 Lab; tolerance0.001. Direct/portable prediction discrepancies are below2e-8. Synthetic copied weights preserve source mass to1.78e-15.

Runtime exited0 after a fresh process scan found no other Python training job. All 111 consumers have 20 warmups and three identity-query timing passes. Selected seed17 settings yield 96 full fits including warmups; four fixed positive mixed probes add16, all exact arrays. Report, static checks and verification run after timing. Final combined numerical suite: 23 tests passed in1.62 s. Earlier primary tests passed before source freezing/fitting. Postprocessing only changed after primary freezing.

Local command notes: an initial `uv run --no-sync ruff` attempt created an unused worktree `.venv` and failed because it had no Ruff; no package installation occurred. All successful work used the explicit existing original-repository Python path above. Ruff found one unused report variable, removed before generating artifacts; no fit code or frozen result was changed. These setup issues did not alter the trained models.

[Protocol](../../research/chromaseed_affine_v1_protocol.md) · [Results](report.md) · [Verification](verification.json) · [Model card](../../architecture/chromaseed_affine_model_card.md).
