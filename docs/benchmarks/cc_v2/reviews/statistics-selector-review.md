# CC v2 statistics selector review — closed

Independent local review, 2026-09-10. Scope: `scripts/cc_v2_statistics_eval.py`, `tests/test_cc_v2_statistics_eval.py`, their called statistics/selector/split helpers, and the estimator selection lock. Script SHA-256: `5b2c2651d0184e5b0fd12e49c2eb61a378723dce1c89e2ead223bf0fbe115a5b`. Test SHA-256: `0a64d08a999a636360525a9e7cf977b26e1d1976b823eb13b03fd3cc17474e47`.

**No actionable finding or blocker identified within this bounded review.** No implementation or source edit was made. All executed fitting/evaluation probes used generated synthetic arrays in temporary directories. The reviewer did not read real target images/GT, run real target predictions, compute real target errors, contact an external review service, or publish anything.

## Source-only fitting and matched controls

The adapter checks the candidate against the source estimator lock and binds that lock to the statistics screen. It checks the exact source data root and screen data hashes. Source predictions are generated with the previously reviewed frozen statistics implementation and verified through candidate, screen/model/script, prediction output, input paths/hashes, and ordered IDs.

`fit()` selects risk and calibration rows before decoding their GT. The underlying split validator rejects overlapping source roles and capture groups. Only risk rows enter head fitting and five-fold `GroupKFold` selection. Ridge's scaler is inside each fold's pipeline; HGB internal early stopping is disabled. The estimator grid, hyperparameters, folds, log-error target, pooled OOF risk at 80% selection and AURC tie-break match the CNN selector. Tied scores use the shared SHA-256-ID ordering. The three feature blocks are separately selected with that same procedure.

Final heads fit all risk rows. Calibration uses only the separate calibration rows and a strictly positive scalar; it does not select the head or alter its ranking. Source calibration scores supply later frozen thresholds. These are empirical calibration and model-selection procedures, with no conditional-risk or distribution-free coverage guarantee. The original CNN and statistics feature representations differ; matching the head search does not imply equal feature dimensionality or identical estimator training cost.

The frozen statistics prediction export computes per-image features for the complete source cache, including its official test images, but reads no GT member and fits no parameters. The adapter subsequently uses only risk/calibration rows. This is inference on unused rows, not joint feature normalization or target-error-based fitting.

## Artifact bindings and evaluation gate

The fitting state binds the adapter, statistics and selector scripts; source implementation identity; source screen and selected base model; estimator lock; source NPZ/manifest; and generated source predictions/sidecar. Bindings are checked again before writing completed selection state. The state records each fitted head's artifact hash, selected candidate, OOF scores, source IDs/folds, calibration scores and scale.

`evaluate()` requires `statistics_selectors[candidate]` in the final head lock to equal the selection JSON hash before any prediction or GT read. It checks source bindings and source identity before processing domains. Each head is hash-verified before loading and before computing its errors. Evaluation inputs and predictions are bound by exact paths, hashes and IDs; inputs, scripts and artifacts are rechecked before the report is written. Modified binding data is rejected rather than silently rebound. The final experiment-wide lock and which candidates are registered in it remain the parent's responsibility.

Source regression labels are restricted to official test indices. External evaluation uses all requested rows and camera-specific slices, with the GT-to-row mapping maintained through sorted indices. The shared refusal helper excludes invalid inputs from every acceptance mask, including nominal full coverage, and labels fallback error summaries as diagnostic. No real-domain result was inspected to establish these properties.

## Independent verification

- `.venv/Scripts/python -m pytest tests/test_cc_v2_statistics_eval.py -q`: **3 passed in 6.19 seconds**. These cover source GT role access, matched grid/group folds, estimator and final locks, changed source inputs, and all-invalid synthetic external refusal.
- `.venv/Scripts/ruff check scripts/cc_v2_statistics_eval.py tests/test_cc_v2_statistics_eval.py`: **all checks passed**.
- Additional reviewer-run synthetic probes accessed exactly 25 risk and 6 calibration GT rows. Replacing all non-risk/calibration prediction, context and cheap-feature rows with NaNs left every head's full recorded result exactly unchanged.
- Independently changed only the synthetic calibration labels. Every candidate's OOF vector, risk/AURC value and selected head remained exactly unchanged. Calibration scales changed as expected; source calibrated mean score equaled mean calibration error within `1e-12`.
- Independently recomputed all 15 synthetic candidate risk-at-80% and AURC values from errors, scores and SHA-256-ID tie ordering; all matched within `1e-12`, and the selected minima matched.
- Installed sentinel wrappers that would fail if prediction or GT reading was reached. Both missing and stale final statistics selector locks raised `Final head lock mismatch` without reaching either wrapper.

Synthetic selected heads were context `ridge100`, cheap `ridge100`, combined `ridge10`. Their calibration scales changed from `1.3886074132`, `1.3797398097`, `1.4796390475` to `3.0039110202`, `2.9847281383`, `3.2008355985` under the calibration-label perturbation. These values document the synthetic probe only and are not real source or target performance evidence.

The first independent metric recalculation used lexical-ID ties and failed its assertion; correcting the reviewer probe to the existing SHA-256-ID tie rule resolved that discrepancy. No production change was indicated. No real-data adapter result was audited here, and no target-based tuning was performed.
