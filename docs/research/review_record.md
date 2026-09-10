# Independent review record — 2026-09-10

Two scoped reviews of the engineering core were performed by an independent coding agent. Findings were checked locally and regression tests were run before and after fixes. This is software review, not independent replication of scientific efficacy.

| Finding | Resolution / evidence |
|---|---|
| Anatomical cheek labels reversed relative to image coordinates | Required explicit mirroring metadata, canonical unmirrored images and anatomical-order masks; test_anatomical_left_is_image_right_unmirrored |
| EXIF geometry mismatch between validator and decoder | Validator uses post-EXIF dimensions; non-square orientation-6 regression |
| NaN/negative uncertainty bound could pass | Finite nonnegative bound required, positive finite tolerance; policy parameter tests |
| Calibration load trusted malformed parameters | Calibration dataclass validates version, provenance, alpha/tolerance/quantile and rejects unsupported domain approval |
| Region bootstrap could be mislabeled image coverage | image_ids required; errors average and scores maximize over regions before resampling/selecting whole images |
| Numeric subject IDs caused NaN subgroup dispersion | Normalize subject identifiers at evaluation boundary; finite summary test |
| Full-resolution images retained for every image | Cache only compact tensors/colors/features; 64×256px preparation exceeded 48MiB before the fix and passes this bounded-memory regression afterward |
| ORT timings could describe a stale model | Check ONNX hash, originating checkpoint hash and dataset binding before benchmark; artifact SHA recorded, tamper regression |

No direct train/test leakage was found in the reviewed fitting path. Remaining limitations are substantive: coarse bbox ROIs, no physical site-registration evidence, untrained model validity, limited synthetic perturbations and no supported real operating domain. These prevent scientific/deployment claims regardless of test count.
