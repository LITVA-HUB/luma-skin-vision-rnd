# Task 1 independent implementation review

Date: 2026-09-10. Scope: `v2.py`, `v2_experiment.py`, `tests/test_cc_v2.py`, the exact task-1 brief, and directly imported split/loss/provenance helpers. **CPU ENGINEERING REVIEW; no real-data training, GPU run, target prediction, or target error evaluation.** No implementation files edited.

Reviewed SHA-256: `v2.py` = `205eba622f6c9707df0c947e8db06cec0708f257c24dffba64385fcb5a071307`; `v2_experiment.py` = `783e57ac90d541c4bf9bbed1d57339af3756fa7819819b388a5ecda2e4892a9c`; `test_cc_v2.py` = `27446039fb71ef038d2a688721928f07e0c905dc1ca3817477730bbfdb4f5a11`.

## Initial verdict: one P2 blocker; resolved in final closure below

### [P2] Execute or enforce the saved inference implementation before exporting frozen predictions

Location: `src/luma_skin_vision/cc/v2_experiment.py:393` (model construction), with the missing check in `verify_run`, lines 314–342.

`verify_run` binds the checkpoint and config and verifies the saved source snapshot bytes, but `predict` subsequently constructs `CompactResidualCC` and the risk feature extractor from the currently imported working-tree modules. It does not compare the live inference implementation with the snapshot or execute the saved implementation. A change to preprocessing, `EPS`, the forward formula, or feature ordering that preserves parameter names can therefore pass all existing binding checks and silently change exported predictions/features for the same checkpoint. Recording a different `prediction_source_identity` at line 421 identifies the discrepancy afterwards but does not enforce frozen-model evaluation.

Independent CPU reproduction: trained a one-epoch synthetic fixture in a temporary directory; replaced the runner's `source_identity` return with a different source hash during `predict`; export succeeded and the manifest contained different training/prediction source hashes. This demonstrates the absent comparison. Separately modifying the stored snapshot was correctly rejected as `Source snapshot changed`.

Requested fix: execute the saved inference implementation, or reject differences in the live files that define model construction, preprocessing and risk features before inference. Comparing the full source hash is sufficient but stricter than necessary if unrelated later analysis files are permitted to change. Preserve a separate explicit, labeled workflow for intentional new inference diagnostics. Add a regression test that modifies an inference-source identity while keeping checkpoint parameter names compatible and verifies refusal (or saved-code execution).

## Independently verified

- `.venv/Scripts/python -m pytest tests/test_cc_v2.py -q`: **20 passed in 3.66s**.
- Focused Ruff check of the three scoped files: **All checks passed**.
- Raw per-channel p1/p6 anchors, quotient-before-clipping, bounded positive residual restoration and projective output normalization implement the stated equivariance on inputs remaining above the floor. The head tests use nonzero weights/biases and verify a prediction different from the anchor. Frozen/eval context and combined features pass three diagonal-gain checks per anchored mode.
- Green-centered log prediction/anchor ratios cancel both channel gains and the prediction's arbitrary common scale. Patch means and standard deviations computed after division by global Gray World are invariant. Column order matches tensor flattening. Direct mode's lack of diagonal invariance is documented.
- Gain augmentation applies identical per-channel gains to image and GT, normalizes GT, and applies common exposure only to the image. Flips and unclipped values are checked. The compatible reproduction-error definition is used.
- Only sorted train/validation rows are retained on the device. Model updates use train rows; best-checkpoint selection uses unaugmented validation; gain stress is validation-only after selection. Risk/cal/test/Sony labels are not used for fitting. The poisoned-test fixture passes. Prediction never accesses the cache's GT array.
- Degenerate channels give finite positive fallback predictions, finite features and explicit `valid=False`; downstream selectors must honor mandatory rejection. Exactness near absolute floors is expressly excluded.
- Checkpoint/config/data fingerprints and saved-snapshot mutation checks work. New training outputs and existing prediction diagnostics are protected from accidental overwrite.

## Documented limitations, not additional blockers

The task brief explicitly requires reuse of existing `official` and `camera` indices. Reading metadata only, the current official cache has 66 capture-date groups shared between source development and official test; the camera protocol has zero such overlaps. `public_protocol_v1.md:11` explicitly describes this unchanged publisher random-image split and forbids interpreting it as scene-disjoint generalization; `cc_v2_plan.md:13` classifies official462 and Sony30 as previously observed regression sets. Preserve those caveats. The source train/validation/risk/calibration groups are checked disjoint in both protocols. No target pixels or errors were inspected for this audit.

The proof begins at the supplied nonnegative cached tensor with fixed masks; it does not establish equivariance of fresh sensor clipping/masking, spectral camera changes, or physical mixed light. Training accepts and reports finite degenerate source inputs rather than discarding them; the source invalid count must be inspected before scientific runs. GPU peak memory, runtime and performance remain unmeasured by this review.

## Final review closure: PASS, no remaining P1/P2 blockers

Root added a strict comparison of `source_identity()["source_hash"]` with the frozen training hash in `verify_run` at `v2_experiment.py:342`, before prediction constructs the model or emits any output. This resolves the reported P2 under the deliberately selected whole-source lock. The hash covers all source Python files plus `pyproject.toml` and `uv.lock`; later changes to those files require restoring the frozen source before inference. Root plans subsequent analysis changes in scripts/docs outside that lock.

The regression test `test_prediction_rejects_changed_live_source` at `tests/test_cc_v2.py:272` trains a synthetic CPU fixture, supplies a mismatching live source hash, and verifies that prediction raises `ValueError` with the expected message. The existing normal prediction test verifies the unchanged-source path still succeeds. Independently reran `.venv/Scripts/python -m pytest tests/test_cc_v2.py tests/test_cc.py -q`: **30 passed in 4.14s**. Focused Ruff: **All checks passed**.

Final reviewed SHA-256: `v2.py` = `205eba622f6c9707df0c947e8db06cec0708f257c24dffba64385fcb5a071307`; `v2_experiment.py` = `a8e03ba8f97521d15401341d0b8df0cf43417cc0d6e89adbbfd73146f4ad44fc`; `test_cc_v2.py` = `f36beff4e625cafba6fa214886479334bab0420d9aa4bed7b20e8d98c53d0ee7`.

This closure concerns implementation and provenance correctness for the planned experiment. It does not establish predictive benefit, calibrated target risk, or GPU resource compliance; those remain measured experiment outcomes.
