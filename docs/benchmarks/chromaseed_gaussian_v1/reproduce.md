# Reproducing and verifying TG

Use the original project's existing `.venv/Scripts/python.exe` from this worktree. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Only the original `data/processed/skin_mskcc_pixels_v1/train.npz` is allowed. No new installations. Never run G/GS verifier mains; their old receipts are immutable inputs.

Execution record (the canonical run is sealed; do not overwrite it):

1. `python -m pytest tests/test_chromaseed_gaussian.py -q` —18 numerical tests passed before primary fitting in2.41 s.
2. `python scripts/chromaseed_feature_groups_verify.py --cache <original-train.npz>` —fresh parent check read-only, session59348 exit0, unchanged receipt459e6554bfba7ce22c63f7377556648632ba07406d20e1d50a42351d8bbf8908.
3. `python scripts/chromaseed_gaussian_train.py run --run experiments/runs/chromaseed_gaussian_v1 --cache <original-train.npz>` —PID26824/session6175, terminal exit0. Nine inner and three final banks,237.595 s excluding imports/audit/runtime.
4. `python scripts/chromaseed_gaussian_audit.py --run experiments/runs/chromaseed_gaussian_v1 --output docs/benchmarks/chromaseed_gaussian_v1 --cache <original-train.npz>` —session99472 exit0,159.493 s.
5. `python scripts/chromaseed_gaussian_runtime.py --run experiments/runs/chromaseed_gaussian_v1 --output docs/benchmarks/chromaseed_gaussian_v1 --cache <original-train.npz>` —session47215 exit0,138.297 s; no concurrent heavy job.
6. `python scripts/chromaseed_gaussian_report.py` —summary, six CSVs, two figures, model contract and next-decision before sealing.
7. `python scripts/chromaseed_gaussian_verify.py --cache <original-train.npz>` —seals once; subsequent invocation validates the same receipt read-only and recursively checks FG/C/H/X/A without rewriting.

Future full replay requires a new run/output directory and separately named orchestration; train/audit/runtime accept the corresponding paths. Canonical report/verifier source binds this run and must not be edited after sealing. Retain the complete old checkpoint/OOF/evaluation archives. The plan and current-status ledgers remain mutable and are not evidence substitutes.

[Protocol](../../research/chromaseed_gaussian_v1_protocol.md) · [Audit](audit.json) · [Timing](runtime.json) · [Receipt](verification.json).
