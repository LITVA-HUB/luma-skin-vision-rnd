# CC V3 source-runner review

Review date: 2026-09-10  
Scope: manual local review of `scripts/cc_v3_experiment.py` and `tests/test_cc_v3_experiment.py`, with read-only inspection of the frozen row reader, metric helpers, model output contract, source-screen lock, source manifest metadata, and prior source frame audit. No GPU work, external upload, or real cache decoding was performed by this review.

## Decision

**PASS for the fixed full 120-epoch mode runs. No unresolved blocker.**

One reporting blocker was found and fixed before the full runs: the epoch-0 history row previously recorded unmeasured training losses as `0.0`. The runner now returns `None` for `train_reproduction`, `train_nll`, and `train_positivity` when `seen == 0`, with a regression test. The already completed two-epoch frame smoke retains its original bytes; its epoch-0 zero fields require the planned erratum and do not affect its validation metrics, validity, or checkpoint choice.

## Scientific and execution checks

- `source_rows()` selects only `train` and `val`, rejects duplicate IDs, requires both roles, and rejects capture-group overlap. The frozen manifest contains 2,234 unique IDs with 1,126 train, 119 validation, 462 test, 259 risk, and 268 calibration rows; train and validation groups are disjoint. The selected source population is exactly 1,245 rows.
- The runner hashes both source files before creating the run directory. Local hashes match the fixed values: `cube.npz` `8323048a...92128` and `cube_manifest.json` `912927e1...7edee`. It passes only the sorted 1,245 train/validation indices to the frozen streaming row reader for `images` and `gt`; test/risk/calibration rows are skipped and never materialized.
- Every run requires a fresh output directory. It snapshots the full `src` tree, `pyproject.toml`, `uv.lock`, and the three executable scripts with per-file hashes, then records exact train/validation IDs, arguments, data hashes, source identity, model shape, optimizer, augmentation, loss weights, framework version, and determinism limitations.
- The exception handler records `failure.json` only when the invocation began with a nonexistent output and created it. It does not alter a pre-existing run directory. Successful runs write `result.json`; failures after directory creation remain distinguishable and retain partial evidence.
- Epoch 0 is evaluated before any optimizer step and is eligible for selection. `validation_key()` permits a checkpoint only for finite all-population validation mean reproduction error with `valid_fraction >= 0.99`; lower-error invalid models cannot win. Strict `<` retains the earliest exact tie.
- On every improvement, `best.pt`, `best_validation.npz`, and `best_metrics.json` are produced from the same in-memory model/evaluation and epoch. The final result hashes that exact `best.pt` and records its byte size and selected epoch; `last.pt` remains separate.
- The point objective equals camera-space reproduction angular error in degrees for positive raw outputs. It uses normalized raw mapped predictions, clips channels only inside the finite invalid-row surrogate, and retains an explicit dimensionless positivity penalty `mean(relu(1e-4 - normalized_raw))`. Negative raw channels receive a finite repair gradient and remain invalid at evaluation; clipping does not alter the model validity mask.
- The canonical mixture NLL is evaluated only on valid frame/GT rows and is weighted by the predeclared `0.02`; the degree-valued point term has weight `1.0` and the normalized positivity term weight `10.0`. Evaluation reports camera-space NLL with the frame Jacobian, reproduction and recovery summaries, validity, transport risk, and per-row best-validation arrays.
- Direct, diagonal, and frame instantiate the same graph and heads with 1,215,535 parameters. Defaults and the frozen source-screen decision give each full comparison 120 epochs, batch 32, seed 17, identical AdamW/cosine schedule, clipping, augmentation, FP32 neural operations, FP64 frame construction, and no AMP/TF32. The two-epoch command is explicitly documented as a feasibility smoke and cannot substitute for the full comparison.
- The prior source frame audit covers exactly the same 1,245 row IDs and fixed data hashes. It found no frame above the `1e6` condition threshold, supporting execution feasibility without turning conditioning into an accuracy claim.

## Verification

`python -m pytest tests/test_cc_v3_experiment.py -q` passed: **4 passed**. The tests cover non-fitting-role exclusion and group separation, raw invalid-point gradients plus positive-row degree geometry, the 99% checkpoint gate, nonfinite selection refusal, and unmeasured epoch-0 training summaries.

## Residual limitations

The runner records CUDA `index_add` nondeterminism and reused development validation in its config. The CLI permits alternate budgets for declared smoke work, so equality of the full three-mode commands remains an orchestration constraint verified from their immutable configs. Abrupt process or host termination can leave a partial run without `failure.json`; normal Python exceptions after output creation are recorded.
