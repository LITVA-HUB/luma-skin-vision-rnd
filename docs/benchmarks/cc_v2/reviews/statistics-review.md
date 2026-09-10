# CC v2 statistics review — closed

Independent local review, 2026-09-10. Scope: `scripts/cc_v2_statistics.py`, `tests/test_cc_v2_statistics.py`, the original and fixed source-screen JSON reports, and the hash-matching local statistics model artifacts. Initial script SHA-256: `ecca55369cb5a65af62e666d5934597f99728f15459c245056fc04edc645f1e5`; fixed script SHA-256: `a989a37d3cab88531f58827bf861fe400b4d13a0ef97b00c61c7b9b07c3c3ac2`. Source hash remained `598a44f190d624db58fd4d2a60368f59bf41b1f3e8620c8c49c55d6e1e680398`.

No implementation edits, target image/GT access, real target prediction, target-error calculation, external review service or publication occurred. Numerical probes use synthetic arrays or saved source-validation predictions/labels only. The original CodeRabbit skill was inspected, but its CLI is absent; the parent explicitly confirmed this bounded local review and did not authorize external service use. These results are not attributed to CodeRabbit.

## Findings found and subsequently closed

### [P2, closed] Make mathematically constant GW features exact before fitting the scaler

Location: `scripts/cc_v2_statistics.py:119–121`, interacting with the `StandardScaler` pipeline at line 213. In GW mode, each channel is divided by its own mean, so the first three `log_minkowski_p1_*` features are mathematically zero. Computing them through maximum scaling and a mean leaves approximately `1e-16` roundoff. The selected, hash-verified `gw_ridge1` model's fitted scaler divides these columns by `[8.206155285e-17, 8.341048759e-17, 8.307413713e-17]`; nonzero Ridge coefficients then turn numerical noise into a learned signal.

Reproduction with synthetic nonuniform positive images and positive diagonal gains: raw GW feature maximum difference was only `4.440892098500626e-16`, but scaled-feature difference reached `4.009273149`; final predictions departed from the required diagonal equivariance by up to **0.1058893477 degrees**. This escapes the existing feature-level `allclose` test. It undermines the claimed invariance of the fitted GW baseline and introduces an artificial feature into its source selection.

Fix: set GW p=1 log features analytically to exact zero (or remove those known constant columns); retain/refit preprocessing and models using source training only, and repeat the unchanged source candidate grid in a new run directory. Preserve the original artifacts. Add an end-to-end fitted-model test asserting prediction equivariance under diagonal gains, rather than only tolerance-based raw-feature equality. Evidence: `statistics-gw-invariance-evidence.json` beside this report.

### [P2, closed] Bind frozen predictions to the input bytes actually read

Location: `scripts/cc_v2_statistics.py:429` and `:447`. `predict_frozen()` reads the image array before prediction, but hashes the NPZ/manifest only after prediction. If either input is regenerated during the operation, the output can describe predictions from old bytes while recording the new input's hash. This defeats the intended reproducibility binding even without GT access.

Synthetic reproduction: replace the two-image input NPZ inside a wrapper around `predict_arrays`, after the function has loaded images A but before it records metadata. Prediction succeeds; its outputs exactly match A, while `input_hashes.npz` exactly equals the replacement B hash. Predictions from B differ by up to `0.8133267444` in a normalized RGB component. Before hash: `0c1df215a7226b8477da34b40e1bd4f3012c843466464524183ffc7ec83810fe`; recorded/replacement hash: `34e558c5494e58537272f7c43df5d9ad29109ebac8c1f48e3f444d6ab02aef5b`. Evidence: `statistics_review_synthetic/reproduction.json` beside this report.

Fix: capture both input hashes before parsing/reading, recheck immediately before publishing prediction artifacts, record the captured hashes and reject changed inputs. Use temporary outputs so a failed binding check does not leave apparently completed artifacts. The same screen/model binding should remain stable during the operation. Add a synthetic mid-prediction input-mutation rejection test.

## Scientific and leakage review

The residual sign and inversion are correct: training uses `log(gt_r/gt_g) - log(anchor_r/anchor_g)` and its blue analogue; reconstruction multiplies the anchor by `exp(residual)` channelwise before unit normalization. Zero residual reproduces the Gray World direction. The fixed learned-output clipping occurs before anchor fusion, so clipping itself does not break the intended diagonal transformation law.

Global statistics and spatial ratios are computed separately for each image. The direct branch retains chromaticity and removes common exposure; its synthetic common-scalar feature difference was at floating-point roundoff. GW features are mathematically diagonal invariant for positive valid anchors and fixed masks, subject to the numerical issue above. An already prepared thumbnail's common mask is part of the input; this review does not assert invariance under sensor clipping or recomputation of masks.

Source screen code selects only train/validation row indices before numerical image/GT decoding. Skipped compressed-cache rows remain uninterpreted bytes. Targets use train GT only, `model.fit` sees train features only, and Ridge's `StandardScaler` is inside the fitted pipeline. HGB early stopping is disabled. Validation is used only for the documented candidate selection. Per-image feature batching does not pool target statistics. Saved records contain 1,126 training IDs and 119 validation IDs, with disjoint IDs and disjoint reported capture groups (44 versus 9). Those cache assignments were not independently reconstructed from real images by this reviewer.

`predict_frozen()` does not read a `gt` NPZ member. It checks the selected local model hash, current/snapshot script hashes and source hash before model loading. All twelve current payloads match the report's model, mode, schema and feature-column declarations. Invalid anchors produce `valid=False` with finite placeholders and an explicit downstream refusal policy; callers must enforce that policy. The function accepts any saved candidate, so the outer experiment freeze must bind which source-selected candidate(s) are eligible before target evaluation.

## Independent verification of the original artifacts

- Reviewer ran `python -m pytest tests/test_cc_v2_statistics.py -q`: **5 passed in 1.98 s**.
- Reviewer ran Ruff on the script and tests: **all checks passed**.
- All twelve model SHA-256 hashes, payload modes/names/schema/columns, current script hash, source hash, and run/report equivalence matched.
- Independently recomputed reproduction error as `acos(sum(gt/pred) / (sqrt(3) * norm(gt/pred)))` using only saved source-validation arrays. Every per-image error matched within `3.36e-11` degrees. Recorded source-validation rankings and both mode winners are correct for these original artifacts.

| Candidate | Independently recomputed source-validation mean, degrees |
|---|---:|
| direct_ridge1 | 3.039677763655 |
| direct_ridge10 | 3.066611776024 |
| direct_ridge100 | 3.175508910394 |
| direct_hgb7 | **2.462731948845** |
| direct_hgb15 | 2.467284297251 |
| direct_extratrees128 | 2.472230025254 |
| gw_ridge1 | **3.335565563660** |
| gw_ridge10 | 3.360493692544 |
| gw_ridge100 | 3.419124744126 |
| gw_hgb7 | 3.429496809681 |
| gw_hgb15 | 3.432237377738 |
| gw_extratrees128 | 3.354935743025 |

These are model-selection diagnostics, not unbiased generalization results. The close direct HGB/ExtraTrees validation scores do not establish a statistically superior model. Independent validation and artifact evidence is saved in `statistics-review-evidence.json`. The reviewer did not fit on or inspect target data.

## Independent fix closure

Both findings were sent to the parent and `/root/review_core`. The implementing agent set valid-channel GW p1 log features to exact zero, preserving invalid fallback values. It captures NPZ/manifest hashes before reading and rechecks them after inference, before creating either output; the captured hashes are recorded in the sidecar. Independent diff inspection confirms both changes.

The reviewer independently ran the complete focused suite after the fix: **8 passed in 2.33 s**, with Ruff clean. The added fitted-Ridge test checks final diagonal-gain prediction equivariance and exact constant-column scaler behavior. Two mutation tests independently replace either synthetic NPZ bytes or manifest IDs during inference and confirm rejection with neither output artifact published. The implementing agent reported that all three added cases failed before its fixes; the reviewer directly reproduced both underlying bugs independently before that implementation.

The unchanged twelve-candidate source grid was rerun into `experiments/runs/ccv2_statistics_fixed/` and `docs/benchmarks/cc_v2/statistics_source_screen_fixed.json`. The reviewer verified unchanged source IDs/groups/counts, source cache hashes, seed, thread count, library versions, validation labels and model hyperparameter grid. **All twelve original model files remain byte-identical**, and all twelve fixed artifacts match their new report hashes and payload declarations.

An independent synthetic gain probe using the newly fitted `gw_ridge1` gives maximum RGB component equivariance error **`3.3306690738754696e-16`**, versus `0.0013236210` before the fix. All three fixed GW Ridge models now have first-three scaler scales exactly one and coefficients exactly zero. Independently recomputed all twelve new validation error vectors; maximum discrepancy remains below `3.36e-11` degrees. Selected candidates remain `direct_hgb7` (**2.462731948845°**) and `gw_ridge1` (**3.325239953382°**), with `direct_hgb7` selected overall. The original GW mean of 3.335565563660° is retained above only as historical evidence.

Full closure evidence: `statistics-review-fixed-evidence.json` beside this report. Both reproduced findings are closed within scope. No further implementation change is requested by this review. The new source-validation scores remain selection diagnostics, and no target-error-based tuning or target-data inspection was performed by the reviewer.
