# Reproducing and verifying FG

Use the original project's existing `.venv/Scripts/python.exe`, not the unused worktree venv. Run from this worktree. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Only the original `data/processed/skin_mskcc_pixels_v1/train.npz` is allowed; its hash and the source/input chain are checked before numerical work. Never invoke G/GS verifier mains: their older receipts are frozen inputs.

The canonical FG run is sealed. The following is its execution record, not a request to overwrite it:

1. `python -m pytest tests/test_chromaseed_feature_groups.py -q` — 14 tests before the primary run.
2. `python scripts/chromaseed_crossfit_verify.py --cache <original-train.npz>` — existing C receipt checked read-only, unchanged.
3. `python scripts/chromaseed_feature_groups_train.py run --run experiments/runs/chromaseed_feature_groups_v1 --cache <original-train.npz>` — primary PID 39764 / session 44863, terminal exit 0.
4. `python scripts/chromaseed_feature_groups_audit.py --run experiments/runs/chromaseed_feature_groups_v1 --output docs/benchmarks/chromaseed_feature_groups_v1 --cache <original-train.npz>` — session 67225, terminal exit 0. The arithmetic count erratum was written before audit and included among its dependencies.
5. `python scripts/chromaseed_feature_groups_runtime.py --run experiments/runs/chromaseed_feature_groups_v1 --output docs/benchmarks/chromaseed_feature_groups_v1 --cache <original-train.npz>` — session 9942, terminal exit 0; no concurrent heavy job.
6. `python scripts/chromaseed_feature_groups_report.py` — builds summary, four CSVs, two figures, report and model contract before sealing.
7. `python scripts/chromaseed_feature_groups_verify.py --cache <original-train.npz>` — seals once; subsequent invocation checks its existing receipt read-only. It also verifies the C/H/X/A chain without rewriting it.

For a future full replay, use a new run and output directory. The train/audit/runtime tools accept `--run` and `--output` as applicable; replay metadata hashes will differ from the canonical records. The current report/verifier deliberately bind the canonical FG paths and must not be redirected by editing their sealed source. Retain this study as the reference and use separately named orchestration if a replay is justified.

[Registered protocol](../../research/chromaseed_feature_groups_v1_protocol.md) · [Arithmetic erratum](../../research/chromaseed_feature_groups_count_erratum.md) · [Audit](audit.json) · [Timing samples](runtime.json) · [Receipt](verification.json).
