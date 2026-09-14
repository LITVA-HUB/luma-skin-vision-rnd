# Reproduce KF and exact KE follow-up

Use fresh run directories and the recorded Python/NumPy/PyTorch/SciPy environment. The source locks deliberately reject resumption with changed code or package versions. Preserve all previous runs. No network or image acquisition is needed; the original prepared TRAIN and previous K banks are required.

```powershell
Set-Location 'C:\Users\dimal\Documents\просто\.worktrees\luma-local-search'
$taskPython = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\.venv\Scripts\python.exe'
$taskCache = 'C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\data\processed\skin_mskcc_pixels_v1\train.npz'
$taskParent = 'experiments/runs/chromaseed_kernel_v1'
$taskRun = 'experiments/runs/chromaseed_fast_kernel_v1_replay'
$taskExact = 'experiments/runs/chromaseed_condensed_exact_v1_replay'
$taskOutput = 'experiments/runs/chromaseed_fast_kernel_v1_replay/report'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

& $taskPython -m pytest tests/test_chromaseed_kernel.py tests/test_chromaseed_kernel_bank.py tests/test_chromaseed_fast_kernel.py tests/test_chromaseed_fast_kernel_protocol.py tests/test_chromaseed_condensed_exact.py -q
& $taskPython scripts/chromaseed_fast_kernel_train.py run --cache $taskCache --run $taskRun --parent-run $taskParent
& $taskPython scripts/chromaseed_fast_kernel_audit.py --cache $taskCache --run $taskRun --parent-run $taskParent --output $taskOutput
& $taskPython scripts/chromaseed_fast_kernel_runtime.py --cache $taskCache --run $taskRun --output $taskOutput
& $taskPython scripts/chromaseed_condensed_exact_run.py --cache $taskCache --source-run $taskRun --run $taskExact --output $taskOutput
& $taskPython scripts/chromaseed_fast_kernel_report.py --run $taskRun --condensed-run $taskExact --output $taskOutput
```

Check each exit code before continuing; a failed audit cannot be skipped. Do not profile alongside another training or memory-intensive benchmark. KF finishes all9 inner banks before locking45 hyperparameter,9 fast-arm and3 size choices; it then fits3 final banks before accessing outer metrics. All4860 analytical configurations reuse bank algebra. 135 models are selected; policy aliases are not additional fits. Optional individual stages: `inner`, `select`, `final`, `evaluate`.

KE starts only after the parent audit and timing have completed. It adds no hyperparameter selection:81 fixed inner and27 fixed final reproduction fits, checking all108 payloads/26 082 rows against exact parent controls. It then times108 standalone fits including warmups and15 synthetic fits including allocation probes. Parent KF timing has180 standalone fits and45 synthetic fits. Replays and repeated timings do not create new hypotheses or independent data.

Measured versions: Python3.12.6, NumPy2.5.3, PyTorch2.8.0+cu128, SciPy1.18.1. SciPy is explicitly declared in the research extra; normal dependency resolution may choose other compatible versions, so match the recorded version for the measured benchmark. `uv lock --offline` updated the lockfile without installing anything. The exact trainer needs SciPy; the frozen inference function needs NumPy. Libraries and generated files do not include a face detector or image pipeline in the20 284B numeric payload.

KF synthetic scaling can take over a minute because the exact8192-row control intentionally measures the costly all-pairs implementation. There are two optional partial timing checkpoints; they do not imply completion. Inspect the actual process/session before resuming anything. An old progress PID alone is insufficient evidence that a run is live.

Current completed sessions: KF training58221/PID36780, audit46267, runtime56039, KE36989, all exit0. Primary run `experiments/runs/chromaseed_fast_kernel_v1`; exact run `experiments/runs/chromaseed_condensed_exact_v1`. No process remains. See [verification](verification.json), [primary protocol](../../research/chromaseed_fast_kernel_v1_protocol.md), [exact follow-up protocol](../../research/chromaseed_condensed_exact_v1_protocol.md), [model card](../../architecture/chromaseed_fast_kernel_model_card.md).
