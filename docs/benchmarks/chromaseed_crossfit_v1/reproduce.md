# Reproduce C

Use the existing original-repository Python environment, not the unused worktree .venv or uv sync. PowerShell from the research worktree:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$cPython='C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$cTrain='C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
& $cPython scripts/chromaseed_crossfit_verify.py --cache $cTrain
```

Existing C/H/X/A verification receipts are read-only. Never run the old G/GS verifier mains or overwrite bound files. For a new reproduction use new directories: crossfit_train.py `run --run NEW_RUN --cache TRAIN`; crossfit_audit.py and crossfit_runtime.py each `--run NEW_RUN --cache TRAIN --output NEW_OUTPUT`. Freeze inherited H settings before fitting all teachers and students. Compare numeric arrays, not timing or ZIP timestamps. The report/verify scripts address the archived study, not arbitrary output paths. Only original TRAIN is required. No new selector or legacy validation/calibration/test access.

[Report](report.md) · [Protocol](../../research/chromaseed_crossfit_v1_protocol.md) · [Verification](verification.json).
