# Reproduce this local experiment

Run from the isolated worktree. Existing Python environment: original repository `.venv`, torch 2.8.0+cu128, NumPy 2.5.3, RTX 4060. Hashes and environment are in source_lock.json. Training uses only the explicit original TRAIN cache. Do not substitute legacy validation or test caches.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $taskPython -m pytest tests/unit tests/test_skin_local_search_core.py tests/test_skin_local_search_protocol.py tests/test_skin_local_search_compact.py tests/test_skin_local_search_precision.py -q
```

For a fresh reproduction, pass NEW explicit run directories. Do not overwrite authoritative runs. The following shows logical stage order; substitute distinct `--run` and corresponding `--source-run` paths for a new run.

```powershell
& $taskPython scripts/skin_local_search_train.py --cache $taskCache --device cuda --stage fit
& $taskPython scripts/skin_local_search_compact.py --cache $taskCache --device cuda --stage fit
& $taskPython scripts/skin_local_search_precision.py --help
```

Prepare all precision payloads before any outer evaluation. Precision `--run` points to the primary run. All selections, final models and manifests must exist and pass hashes before the following evaluation stages. A replay on these already exposed cohorts never becomes independent confirmation.

```powershell
& $taskPython scripts/skin_local_search_precision.py --stage prepare
& $taskPython scripts/skin_local_search_train.py --cache $taskCache --device cuda --stage evaluate
& $taskPython scripts/skin_local_search_compact.py --cache $taskCache --device cuda --stage evaluate --acknowledge-outer-evaluation
& $taskPython scripts/skin_local_search_precision.py --cache $taskCache --stage evaluate
& $taskPython scripts/skin_local_search_report.py
& $taskPython scripts/skin_local_search_supplement.py
```

The report/supplement currently target the authoritative directories as constants; point their RUN/OUT constants to a fresh replay when regenerating its report. The supplement additionally expects the separate runtime_device_audit.json and historical_reference_audit.json receipts in the primary run; those are verification artifacts, not training dependencies. Random seeds are fixed, but strict CUDA deterministic-algorithm mode was not enabled: bit-identical stochastic training across stacks is not promised. Stored artifacts are SHA-bound, and rerun quality comparisons should use tolerances and the full seed set.

The compact study has a documented evaluation-bookkeeping amendment. Preserve `source_lock_fit_archive.json`, `source_lock_amendment.json` and immutable `frozen_prefixes.json` with that history. Do not alter a frozen file just to update completion status. Numerical models and selections were unchanged by this amendment.
