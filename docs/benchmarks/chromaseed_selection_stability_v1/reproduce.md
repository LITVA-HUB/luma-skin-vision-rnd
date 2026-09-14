# Reproduce ChromaSeed-S

Use the existing W/P runs and a fresh diagnostic output directory. Original TRAIN only. This sequence loads three metadata/target arrays and nine fixed inner OOF archives; it never retrains a model or reads outer predictions/results. The W verification receipt is a fixed provenance input, so preserve it.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskRun = 'experiments/runs/chromaseed_selection_stability_v1_replay'
$taskOutput = 'experiments/runs/chromaseed_selection_stability_v1_replay/report'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed_selection_stability.py -q
& $taskPython scripts/chromaseed_selection_stability_run.py --cache $taskCache --run $taskRun
& $taskPython scripts/chromaseed_selection_stability_audit.py --cache $taskCache --run $taskRun --output $taskOutput
& $taskPython scripts/chromaseed_selection_stability_report.py --run $taskRun --output $taskOutput
```

Check each exit code before the next command. Eight numerical tests passed in 0.14 seconds before the original real calculation. Frozen code reproduces all 891 original candidate scores to 1.78e-15. The primary diagnostic returned exit 0 directly (PID13944); first audit session56446 returned exit 0. The audit was then tightened to check additional reported scalar fields and remove an unused counter; final session30639 also returned exit 0. The final audit checks 12,474 person/candidate values, all 300,000 bootstrap choices/ranks and 210 deletion choices with independent aggregation/selection loops. It shares the verified CIEDE2000 formula and frozen role helpers.

60,000 person resamples are shared across five families per role. Do not count 300,000 decisions as independent samples or new model fits. The figures' empirical ranges omit fitting/fold uncertainty and must not be presented as confidence in new-phone accuracy. No patient identifiers are exported: score axes are ordinal and local.

[Protocol](../../research/chromaseed_selection_stability_v1_protocol.md) · [Report](report.md) · [Verification](verification.json).
