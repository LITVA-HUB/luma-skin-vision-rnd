# Reproducing and verifying NR

Use the original project `.venv/Scripts/python.exe` from this worktree. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 and OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1. Only original data/processed/skin_mskcc_pixels_v1/train.npz. No installation, no concurrent heavy jobs. Never G/GS verifier mains; TG/FG/C/H/X/A reruns are read-only.

Execution record, not an instruction to overwrite sealed canonical artifacts:

1. `python scripts/chromaseed_gaussian_verify.py --cache <original-train.npz>` —fresh read-only parent, session59007 exit0.
2. `python -m pytest tests/test_chromaseed_neural_readout.py -q` —24 tests before primary,1.47 s. New test batch misuse corrected after checking actual TG consumer; erratum retained separately. Independent QR and representation reference tests pass.
3. `python scripts/chromaseed_neural_readout_train.py run --run experiments/runs/chromaseed_neural_readout_v1 --cache <original-train.npz>` —PID18004/session82004, exit0,61.634 s.
4. `python scripts/chromaseed_neural_readout_audit.py --run experiments/runs/chromaseed_neural_readout_v1 --output docs/benchmarks/chromaseed_neural_readout_v1 --cache <original-train.npz>` —initial79371 exit1 at an overly strict exact-count assertion after all numerical comparisons; corrected only the audit to record exactness under unchanged registered tolerance. Full repeat11895 exit0,141.416 s.503/504 bases bitwise, one w1 difference7.45e-9; all3024 QR refits pass.
5. `python scripts/chromaseed_neural_readout_runtime.py --run experiments/runs/chromaseed_neural_readout_v1 --output docs/benchmarks/chromaseed_neural_readout_v1 --cache <original-train.npz>` —session51100,378 complete fits including126 warmups.
6. `python scripts/chromaseed_neural_readout_report.py` —generates summary/five CSVs/two figures/model card/next decision before sealing.
7. `python scripts/chromaseed_neural_readout_verify.py --cache <original-train.npz>` —seals once, then reruns read-only without rewriting old receipts.

Future numerical replay requires separate run/output paths and separately named orchestration. Canonical report/verifier bind this series and must not be edited after sealing. Preserve the original primary/model/OOF/prediction archives and every control. Mutable plan/status ledgers are not immutable evidence.

[Protocol](../../research/chromaseed_neural_readout_v1_protocol.md) · [Audit](audit.json) · [Timing](runtime.json) · [Receipt](verification.json) · [TG erratum](../../research/chromaseed_gaussian_consumer_erratum.md).
