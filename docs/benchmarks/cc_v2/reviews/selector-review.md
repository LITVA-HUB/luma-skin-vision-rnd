# CC v2 selector review — closed

Scope: independent inspection of root-authored `scripts/cc_v2_select.py` and additive synthetic tests in `tests/test_cc_v2_select.py`. The reviewer changed only that test file and this requested report. No real target/model evaluation, GPU experiment, or production-script/source edit was performed by the reviewer.

## Scientific fitting review

Source risk and calibration rows are isolated before errors are computed. Date-group five-fold CV fits `StandardScaler` inside each training fold, disables HGB internal early stopping, and uses no calibration/test labels in candidate fitting or selection. Each context/cheap/combined block receives the same ridge/HGB candidate grid and folds. Final selected heads fit all risk rows; a separate calibration set supplies a positive scalar and source thresholds. These are empirical risk estimates, not conditional or distribution-free guarantees. Pooled CV selection scores remain source model-selection diagnostics, not independent target performance evidence.

## Findings and closures

1. **Prediction/checkpoint provenance:** fitting previously accepted modified predictions with matching IDs. Root added prediction-manifest checks binding checkpoint, config, source identity, output file hashes, and input NPZ/manifest hashes. Selector state binds the immutable prediction manifest.
2. **External evaluation input binding:** Sony/external inputs could previously change while retaining IDs. Hash and exact input-path checks now reject a changed or substituted evaluation cache/manifest.
3. **Source cache binding at fit:** initial provenance repair still allowed a prediction manifest pointing to a different, internally hash-consistent input cache. The regression failed with `DID NOT RAISE`. Root added an exact source NPZ/manifest path check against the supplied dataset before fitting. Narrow closure inspection confirms that check is in `fit()` before row/error use.
4. **Mandatory refusal:** a large numerical score alone did not prevent invalid rows from entering full-coverage acceptance. Every frozen acceptance mask now intersects validity. Diagnostic curves use valid rows, expose maximum attainable population coverage, and mark impossible requested coverage as unattainable. Full-population fallback error summaries are explicitly diagnostic.
5. **Zero-error calibration:** scale zero collapsed ranking despite the positive-scaling contract. Root added a positive numerical floor, preserving ranking in the zero-error calibration edge case.

## Verification evidence

The eight synthetic tests cover: poisoning every non-risk/cal GT/prediction/feature without changing source fitting results; disjoint capture-date folds and matched candidate grids; modified prediction output and wrong checkpoint rejection; different source cache rejection; external input hash rejection; all-invalid and mixed-validity acceptance; exact accepted-subset metric aggregation; and positive scale for perfect calibration targets.

- Reviewer observed the remaining source-cache regression fail before its fix: **7 passed, 1 failed**, with the failure `DID NOT RAISE ValueError`.
- Parent reported the complete focused suite after the fix: **8 passed in 10.63 s** (`tests/test_cc_v2_select.py`). This is parent-run evidence, not a second full-suite run by this reviewer.
- Reviewer independently re-ran the specific source-cache closure test after inspecting the fix: **1 passed, 7 deselected**.
- Reviewer ran Ruff on the owned test file: **All checks passed**.

All reported findings are closed within this scope. No further source implementation changes are requested. Real-data metric review is a later, separate review; these synthetic fixtures establish engineering behavior only.
