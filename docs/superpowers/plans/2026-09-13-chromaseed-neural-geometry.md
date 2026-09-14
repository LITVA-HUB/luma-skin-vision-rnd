# Neural geometry implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task inline. No delegation is authorized.

**Goal:** Determine from original inner data whether a single stronger-penalty readout experiment is justified, while preserving the full active goal.

**Architecture:** A Schur-complement calculation on fixed NR bases is checked by a separately constructed, whitened and intercept-residualized SVD. The runner reconstructs exact parent row roles and re-scores saved inner predictions, then writes all spectra and a decision without opening outer results.

**Tech Stack:** Existing Python3.12/NumPy/SciPy environment, one CPU thread, pytest/ruff. No installation.

**Spec:** docs/research/chromaseed_neural_geometry_v1_protocol.md.

## Global constraints

- Original TRAIN only; no exposed validation/test or new data.
- 9 inner banks,378 bases,756 designs,3780 df values;252 existing candidate scores.
- Existing NR and prior sources/receipts are immutable. Plans and status documents remain mutable.
- Keep full goal active; geometry is not photo-quality evidence.

## Task1: numerical geometry and independent oracle

Files: create scripts/chromaseed_neural_geometry.py, scripts/chromaseed_neural_geometry_reference.py, tests/test_chromaseed_neural_geometry.py.

Interfaces: spectrum(z,weights,metric=None)->(eigenvalues,bias_dimension,multiplicity); degrees(spectrum_tuple,alpha)->total df. source_design(source,x) preserves parent arithmetic; source_metric(source,y,weights) returns the normalized tensor.

- [x] Add direct full-hat-trace tests for both objectives, intercept-only/large-alpha limits and feature-shift invariance.
- [x] Run explicit Python -m pytest tests/test_chromaseed_neural_geometry.py -q; observe absent implementation fail.
- [x] Implement Gram/Schur/eigenvalue calculation and independent QR/SVD reference with analytic tensor.
- [x] Run the tests and ruff on only the new files.

## Task2: frozen inner-only calculation

File: create scripts/chromaseed_neural_geometry_run.py. It consumes parent NR bases/receipts/OOF/selection, source manifest and read-only verification; writes source_lock.json, spectra.npz, results.json and workflow.json under experiments/runs/chromaseed_neural_geometry_v1.

- [x] Bind consumed files and numerical sources before computing; reject wrong TRAIN or altered parent hashes.
- [x] Reconstruct fit/query indices with roles/folds_for and assert parent row hashes and person separation.
- [x] Compute all756 spectra and3780 df values, independently validate each; reconstruct252 scores from all9 OOF banks.
- [x] Poll the original live handle to terminal and inspect coverage plus the predeclared decision.

## Task3: decision and evidence

Files: create docs/benchmarks/chromaseed_neural_geometry_v1/report.md and verification.json; update mutable goal/status/AGENTS after verification.

- [x] Report geometry and old-curve evidence without adding or selecting outer results.
- [x] Verify new tests, all source/input/output hashes and relative local links; record exact terminal process evidence.
- [x] If both registered criteria hold, preregister one stronger-alpha readout study before fitting it. If they fail, document why and pursue a different bounded learning question.

No commit or publication is part of this plan. Execution continues inline under the user's autonomous authorization.

## Execution evidence

NG completed:9 tests pass;756 independent geometry checks,3780 df values and252 exact old scores; primary PID34636 exit0. Receipt57b1d6b00b51e9f761ad34b2062f28a1186cd403f0ba40a52e24b0d105014e25; read-only replay passed. Both preregistered criteria met and NS was registered and completed.
