# Synthetic engineering baseline report — 2026-09-10

**SYNTHETIC ONLY. Not real skin accuracy, technological advantage or a passed scientific gate.**

Measured data volume: 60 toy subjects, 480 JPEG patch images, 960 region records; subject split 24/12/12/12. Two synthetic camera-gain settings and two synthetic light-gain settings are not real smartphone/light pipelines.

All six methods use identical subject splits and complete paired cheek records. Learned models ran two epochs at 64px from random initialization. This is intentionally a pipeline check, not a tuned model comparison. Requested 80% is rounded down to a common whole-image count; actual coverage is shown. Errors average both cheeks; selectors retain entire photos.

| Method | Mean ΔE00 | Requested 80% risk | Actual coverage | Accepted images |
|---|---:|---:|---:|---:|
| baseline_a0 | 3.0323 | 3.1776 | 79.1667% | 76 |
| baseline_a1 | 6.9589 | 7.0466 | 79.1667% | 76 |
| baseline_a2 | 2.0270 | 2.0711 | 79.1667% | 76 |
| baseline_c | 26.2852 | 25.7345 | 79.1667% | 76 |
| baseline_c_plus | 26.1407 | 24.3790 | 79.1667% | 76 |
| proposed_v1 | 24.4760 | 22.1875 | 79.1667% | 76 |

A2 is numerically strongest on this toy generator. The proposed neural prototype does not beat the strongest classical baseline here. This says nothing conclusive about real-data potential, and no architecture/config was tuned to make the comparison look successful. Gray World is worse than no correction on skin-only toy patches, consistent with its violated scene-average assumption.

Raw provenance: [synthetic_smoke_evidence.json](synthetic_smoke_evidence.json). Full curve: [plot](synthetic_risk_coverage.png). Each experiment keeps its OOF membership audit, model hash, frozen calibration and test predictions.
