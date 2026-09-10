# Independent legacy comparator review

Date: 2026-09-10. Scope: `scripts/cc_v2_legacy.py` and directly used selector, v1 feature/metric and provenance helpers. **STATIC AND SYNTHETIC CPU REVIEW ONLY.** No real target predictions/errors, real risk-head fitting, GPU operations or source-code edits were performed.

## Initial verdict: two P2 blockers; resolved in final closure below

### [P2] Authenticate the original v1 artifacts before fitting a new selector

Location: `scripts/cc_v2_legacy.py:27–36`, followed by `:76–78`.

`verify_legacy` compares current data with the supplied config and four source files with commit `7637d6d`, but never verifies that the supplied checkpoint/config/risk states are the original run's artifacts. `fit` records their current hashes only after loading and prediction. A swapped same-architecture checkpoint, altered protocol/config or modified original risk head can therefore become the newly recorded “legacy” baseline and pass later evaluation binding. This does not establish reproduction of the preserved v1 evidence. In a synthetic check, the verifier accepted an unregistered run with arbitrary checkpoint bytes and an unbound source hash; no checkpoint was loaded in that check.

Fix: look up the exact allowed run in an immutable version of `docs/benchmarks/public_evidence_manifest.json`, compare the checkpoint SHA-256 and data hashes, and verify config/risk-head bytes against the corresponding committed `docs/benchmarks/public_runs/<run>/` records before model loading or new output creation. Preserve the existing between-fit-and-evaluate checks as well. Add a synthetic test that substitutes a compatible checkpoint or original risk state and confirms rejection before prediction. A newly recorded hash alone is insufficient to authenticate a historical run.

### [P2] Enforce the fitted selector's script and helper identities before evaluation

Location: `scripts/cc_v2_legacy.py:132–139`; stored but unchecked identities at `:86–87`.

The evaluation preflight checks checkpoint/config/original-head/joblib hashes, but ignores the recorded `script_sha256` and `helper_sha256`. The live `cc_v2_select.py` supplies `raw_predict` and `selective_result`; changing clipping, scaling, feature/metric behavior or refusal logic after fitting can change exported baseline scores and results. `source_identity` excludes the scripts directory, and the evaluation report does not separately record the live helper hash, so its current provenance cannot reconstruct which helper was executed.

Fix: compare both live script hashes with their fit-time values before model loading or any evaluation input access. If intentional post-fit evaluation-code changes are supported, give that workflow a separately frozen and explicit identity. Synthetic reproduction changed the two stored expected hashes and evaluation still reached a deliberately injected `model_load` sentinel; it was stopped there, before model loading, external access or any error calculation.

## Verified behavior

- `.venv/Scripts/python -m pytest tests/test_cc_v2_select.py -q`: **8 passed in 9.21s**. Focused legacy-script Ruff: **All checks passed**.
- Independent legacy-fit fixtures used 20 risk rows across five distinct groups and five separate calibration rows. Poisoning all other source image/expert/GT rows and the Sony fixture with NaNs produced identical selected models, OOF candidate records, calibration scores and scales. The prediction stub asserted exactly 25 retained rows and CPU tensors. No estimator parameters were trained.
- Within risk training, `GroupKFold(5)` keeps each capture group in one validation fold. Standardization for ridge is fitted inside each fold's pipeline; final heads fit only risk rows. Calibration uses separate cal rows and a strictly positive multiplicative scalar. Both feature blocks use the same five candidate configurations and source-only selection rule as the new-CNN selector.
- The inherited split function is used consistently. Historical source train/validation/risk/calibration separation relies on the original config/data being authenticated; the first blocker must be fixed to make that reliance reliable. The known official random-image test overlap remains an explicitly documented limitation, not a new legacy implementation issue.
- Old-head feature dispatch preserves `context`, `disagreement` and `combined` behavior; classical estimates use the model context and the combined feature block as in v1. Saved JSON lists are restored to arrays. The inherited additive-offset/standardization/clipping score equation was independently checked on synthetic features in all three modes and through JSON serialization.
- Old-head source calibration scores are recomputed from the frozen old states, without fitting new offsets or consulting target labels. The new heads' source calibration scores are stored at fit time. `selective_result` enforces mandatory invalid-input refusal in fixed-threshold masks and attainable selective curves; the existing mixed/all-invalid synthetic tests pass. Full-population fallback angular summaries remain diagnostics.
- Output-existence checks protect the original run artifacts. The evaluated legacy model is not retrained. The intended old-head numerical parity with preserved source/Sony reports still needs checking after the authorized global configuration lock; no real regression evaluation was run for this review.

Reviewed SHA-256: `scripts/cc_v2_legacy.py` = `3bb7f8dd573d88dd4b7dcd7fe26a4e61e3c36bc7cd0b021b502e7fec247917da`; `scripts/cc_v2_select.py` = `cea9e3b55e7a887766fc9cee79346390cd5197fc666d3121134852946bc15686`.

## Final review closure: PASS; no remaining P1/P2 blockers

Root corrected both issues without changing the estimator core. `verify_legacy` now authenticates the exact run and model SHA-256 using the evidence manifest from immutable commit `7637d6d`, and compares config and old risk-head bytes with that commit's preserved per-run records before model loading or fitting. Only CRLF/LF representation is normalized for those JSON comparisons; values, keys and other bytes must remain identical. The existing source-file and dataset checks remain in place. `evaluate` now enforces both fit-time script/helper hashes at lines 147–150 before model construction and evaluation cache loading.

Independent synthetic preflight checks passed:

- Registered original CRLF JSON accepted against the immutable LF fixture.
- Changed checkpoint, config and risk-head bytes each rejected.
- Unknown run rejected.
- Script and helper identity mismatches each rejected with zero model-loader calls.
- Matching identities passed the guards and were deliberately stopped at a model-loader sentinel before any model, data or target access.

The immutable Git responses and dataset hash lookup were mocked with synthetic values for these checks; no historical checkpoint was loaded and no real dataset values were read. No fitting or evaluation was performed during closure. Focused Ruff still passes. The prior source-only fitting/selector tests remain applicable because those algorithms did not change.

Final reviewed SHA-256: `scripts/cc_v2_legacy.py` = `3a1d6b89a689097179dfd650d71dc12126bcf604171c6e72dd787ae2fd4c919a`; `scripts/cc_v2_select.py` = `cea9e3b55e7a887766fc9cee79346390cd5197fc666d3121134852946bc15686`.

Review clearance permits the planned source-only confidence fitting. Actual old-head regression parity and comparative target results remain post-lock measurements, not conclusions from this engineering review.
