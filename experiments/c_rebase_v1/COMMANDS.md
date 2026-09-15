# Executed commands and reproduction

Run from repository root. This experiment uses the published, verified feature cache; no participant photographs or network calls are needed during training. Python 3.12.14, CPU torch 2.8.0+cpu; complete recipe and preprocessing are immutable in `config.lock.json` and `freeze.lock.json`.

## Environment actually used

```bash
python -m venv --system-site-packages .venv-c-rebase
.venv-c-rebase/bin/python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
.venv-c-rebase/bin/python -m pip install pytest==8.4.2
.venv-c-rebase/bin/python -m pytest tests/test_c_rebase_v1.py scripts/c_rebase_v1/test_diagnose.py -q
```

Result: 7 tests passed. The runtime already contained NumPy, scipy, scikit-learn and Pillow at versions in `requirements.txt`. A new machine can create a Python 3.12.14 environment and install that requirements file; this session did not independently test package installation on macOS. The trainer refuses mismatched Python/PyTorch/NumPy/sklearn versions.

## Fixed six-fold training — executed

```bash
.venv-c-rebase/bin/python scripts/c_rebase_v1/train.py \
  --config experiments/c_rebase_v1/config.lock.json \
  --mapping experiments/c_rebase_v1/person_folds.lock.json \
  --features docs/benchmarks/mskcc_error_floor_audit_2026_09_15/data/capture_features_anonymized.npz \
  --freeze experiments/c_rebase_v1/freeze.lock.json \
  --output experiments/c_rebase_v1/run --fold all \
  > experiments/c_rebase_v1/training_stdout.log 2>&1
```

For another reproduction, replace only `--output` and the log path with unused paths. The output must not exist. Do not regenerate the mapping or config. Checkpoints save full model, optimizer, normalization and RNG state at epochs 50 and 100. No checkpoint is supplied to this trainer: all folds start from initialization.

OOF is `run/experiment/private_review/c_rebase_v1_oof.npz`; an exact local copy also exists at repository-root `experiment/private_review/c_rebase_v1_oof.npz`. Public OOF contains only pseudonymous IDs and diagnostics. SHA256: `f0b492bc17087c86359f9c05527fbf01863651c9d6bd3b444cc8bc52cdf0affa`.

## Independent fold reproduction — executed

Before repeating, the original `run/fold_0` was moved to `run/reproduction_quarantine/fold_0_original`. It was preserved, not used as a checkpoint input. The move is recorded in `run/reproduction_move_receipt.json`.

```bash
.venv-c-rebase/bin/python scripts/c_rebase_v1/train.py \
  --config experiments/c_rebase_v1/config.lock.json \
  --mapping experiments/c_rebase_v1/person_folds.lock.json \
  --features docs/benchmarks/mskcc_error_floor_audit_2026_09_15/data/capture_features_anonymized.npz \
  --freeze experiments/c_rebase_v1/freeze.lock.json \
  --output experiments/c_rebase_v1/reproduction_fresh --fold 0 \
  > experiments/c_rebase_v1/reproduction_stdout.log 2>&1

.venv-c-rebase/bin/python scripts/c_rebase_v1/verify_reproduction.py \
  --original experiments/c_rebase_v1/run/reproduction_quarantine/fold_0_original \
  --repeat experiments/c_rebase_v1/reproduction_fresh/fold_0 \
  --output experiments/c_rebase_v1/reproduction_result.json
```

Result: 155 holdout rows / four people, predictions bitwise equal, maximum absolute Lab difference 0, model tensors at both checkpoints equal, normalization and trajectory logs equal. No numerical fallback used. The original output is retained under quarantine.

## Error-floor audit — executed

```bash
PYTHONPATH=src .venv-c-rebase/bin/python scripts/c_rebase_v1/diagnose.py \
  --oof experiments/c_rebase_v1/run/experiment/private_review/c_rebase_v1_oof.npz \
  --rows ../private_artifacts/error_floor/audit_train_rows.private.json \
  --capture docs/benchmarks/mskcc_error_floor_audit_2026_09_15/data/capture_features_anonymized.npz \
  --capture-anonymized \
  --capture-report docs/benchmarks/mskcc_error_floor_audit_2026_09_15/data/capture.json \
  --instrument docs/benchmarks/mskcc_error_floor_audit_2026_09_15/data/instrument.json \
  --fixed-masks docs/benchmarks/mskcc_error_floor_audit_2026_09_15/data/diagnostic_subset_masks.anonymized.npz \
  --diagnostic-lock experiments/c_rebase_v1/diagnostic_plan.lock.json \
  --output experiments/c_rebase_v1/run/audit \
  > experiments/c_rebase_v1/audit_stdout.log 2>&1
```

The initial analysis invocation used the base Python with identical NumPy/scipy versions; that output was preserved privately and then the command above was rerun in the frozen environment. No model/diagnostic recipe changed.

Original image/metadata restoration and capture extraction commands remain in `docs/benchmarks/mskcc_error_floor_audit_2026_09_15/COMMANDS.md`. They are earlier completed work, not repeated model training. The original patient IDs are deliberately absent from public Git. The exact 24-person private list is saved separately for the owner; public mapping contains all 24 aliases, rows and its SHA256.

### Public-only audit reproduction — also executed

The same command was run with `--rows experiments/c_rebase_v1/train_rows.anonymized.json` and an unused output directory. All numerical JSON content matched except the expected provenance differences; top50/strata/correlation CSV files matched byte-for-byte and all diagnostic NPZ arrays matched exactly. See `anonymized_audit_reproduction.json`. A clone can use:

```bash
PYTHONPATH=src .venv-c-rebase/bin/python scripts/c_rebase_v1/diagnose.py \
  --oof experiments/c_rebase_v1/run/experiment/private_review/c_rebase_v1_oof.npz \
  --rows experiments/c_rebase_v1/train_rows.anonymized.json \
  --capture docs/benchmarks/mskcc_error_floor_audit_2026_09_15/data/capture_features_anonymized.npz \
  --capture-anonymized \
  --capture-report docs/benchmarks/mskcc_error_floor_audit_2026_09_15/data/capture.json \
  --instrument docs/benchmarks/mskcc_error_floor_audit_2026_09_15/data/instrument.json \
  --fixed-masks docs/benchmarks/mskcc_error_floor_audit_2026_09_15/data/diagnostic_subset_masks.anonymized.npz \
  --diagnostic-lock experiments/c_rebase_v1/diagnostic_plan.lock.json \
  --output experiments/c_rebase_v1/audit_reproduced
```

## Boundaries

This is an analytical experiment, not a replacement face-inference package. No production C weights, parser, API, thresholds or inference behavior were changed. No model search or error predictor was trained. No reserved source-validation/calibration/test was evaluated. Oracle and privileged exclusions are `LEAKY_DIAGNOSTIC_ONLY`, never deployable selection rules.
