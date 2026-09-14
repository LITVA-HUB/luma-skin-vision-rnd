# Reproduce H locally

Use the existing original-repository Python3.12 environment, NumPy2.5.3/SciPy1.18.1; do not run uv sync or alter older frozen dependency files. Work from the isolated research worktree. PowerShell, one CPU thread:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$hPython='C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$hTrain='C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
& $hPython scripts/chromaseed_hybrid_verify.py --cache $hTrain
```

That verifier is read-only when its receipt exists and recursively uses the read-only X/A paths. Never run the old G/GS verifier mains; they rewrite now-bound receipts. The audit/runtime/report commands below target a separate output directory if repeated. Do not overwrite bound archived reports or numeric run files.

For a new reproduction run use the frozen scripts, a new --run directory and --output directory: train script `run --run NEW_RUN --cache TRAIN`, audit script `--run NEW_RUN --cache TRAIN --output NEW_OUTPUT`, then runtime with those same arguments. Freeze9 inner banks before selection and3 final banks; the runner enforces this order. Compare numeric payloads and predictions, not timing bytes or ZIP timestamps. No legacy held-out dataset is needed. The report/verify scripts address the original recorded study and are integrity checks, not general new-run report generators.

[Protocol](../../research/chromaseed_hybrid_v1_protocol.md) · [Report](report.md) · [Verification](verification.json).
