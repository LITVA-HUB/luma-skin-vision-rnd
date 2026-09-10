# CC V3 source-screen results review

Review date: 2026-09-10  
Scope: read-only audit of the three completed `ccv3_{direct,diagonal,frame}_e120_s17` runs, `scripts/cc_v3_source_report.py`, and the generated `docs/benchmarks/cc_v3/source_screen` evidence. This review read only the frozen source validation GT and expert rows needed for independent recomputation. It did not access source test, risk, calibration or INTEL-TAU GT, run a model, or use the GPU.

## Decision

**PASS — no receipt, metric, risk80, checkpoint, code-snapshot or budget inconsistency found.**

The matched direct graph is the source-screen winner. The diagonal control is worse, and the full color-frame hypothesis is substantially worse on the reused development validation. This is negative evidence for the proposed frame mechanism at the frozen source budget; it is not an independent-test, calibrated-risk or novelty result.

## Independently recomputed table

| Method | Best epoch | Valid at best | Reproduction mean | Uncalibrated risk80 mean | Exact risk80 coverage | AURC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph direct | 29 | 100% | 2.546505° | 2.304012° | 95/119 = 79.832% | 1.759400 |
| Graph diagonal | 69 | 100% | 2.941570° | 3.166782° | 95/119 = 79.832% | 3.421570 |
| Graph frame | 54 | 100% | 4.281256° | 4.154873° | 95/119 = 79.832% | 3.732887 |

The source report's classical all-coverage reproduction means also recompute as Gray World 4.466948°, Max RGB 5.337211°, Shades of Gray 4.247940°, and Gray Edge 4.390398°. Those estimators have no transported-posterior risk score in this artifact, so no classical risk80 comparison is implied.

`risk80` follows the existing fixed-coverage convention: sort by ascending raw transported-risk score with the deterministic ID-hash tie break and retain `floor(119 × 0.80) = 95` rows. The label should therefore remain accompanied by its realized 79.832% coverage. Diagonal risk80 is worse than its own all-coverage mean, so its raw proxy has harmful ordering at this operating point. Frame ordering helps slightly relative to its own full mean but remains far worse than direct. All risk curves are uncalibrated and use the same reused validation set that selected the checkpoint.

## Run and checkpoint correspondence

- Each history contains exactly epochs 0–120. Epoch 0 has `None` for all unmeasured training summaries; epochs 1–120 contain measured summaries.
- The independently selected minimum finite validation-reproduction epoch among rows with at least 99% valid predictions is 29/69/54 for direct/diagonal/frame. Each agrees with `result.json`, `best_metrics.json`, and the saved best-validation arrays.
- Direct and diagonal remain 100% valid throughout. Frame's minimum history validity is 118/119 = 99.1597%, still eligible under the frozen threshold; its selected epoch is 100% valid. The report does not hide fallback rows.
- Each local `best.pt` SHA-256 equals its result receipt. Each `best_validation.npz` contains 119 predictions, validity flags, transported risks, invalid-posterior masses, frame conditions, reproduction/recovery errors and camera NLL values with matching first dimensions.
- Independent FP64 `atan2` recomputation from the saved predictions and the runner-equivalent FP32 validation labels differs from saved recovery/reproduction arrays by less than `3e-11` degrees. All independently recomputed summary statistics agree with `best_metrics.json` within `1e-6`.
- The five copied evidence files per mode exactly match the corresponding immutable run files and the SHA-256 values in `summary.json`. The summary validation IDs exactly match the 119 frozen manifest validation IDs. The current report script hash matches the hash recorded in the summary.

## Matched code and compute budget

All modes used the same 1,126 training IDs, 119 validation IDs, frozen cache/manifest hashes, seed 17, 120 optimizer epochs, batch 32, CUDA device, augmentation, AdamW/cosine schedule, gradient clipping, loss weights and executable script hashes. Only `mode` and the fresh output directory differ. Each result records exactly 1,215,535 trainable parameters.

| Mode | Recorded elapsed time | Peak allocator memory including source cache | Recorded source cache |
| --- | ---: | ---: | ---: |
| Direct | 114.780 s | 643.093 MiB | 233.452 MiB |
| Diagonal | 116.492 s | 643.093 MiB | 233.452 MiB |
| Frame | 123.265 s | 643.093 MiB | 233.452 MiB |

The small runtime difference is consistent with deterministic frame construction overhead and does not represent extra epochs, data or learnable capacity. The report script verifies data hashes, validation identities, code snapshots, checkpoint hashes, complete histories and comparable arguments before creating its fresh output directory. Its generated title and scope correctly say matched 1.22M graph architectures, 1,126 training/119 reused-development validation images, and one seed.

## Interpretation and limits

The direct graph beats the four saved classical full-image expert baselines on this reused validation set, while the frame result is approximately tied with the strongest classical mean rather than improving on direct. Because architecture choice and risk ordering are both examined on the same reused 119-image development validation set, these numbers support only source feasibility and hypothesis revision. They do not establish generalization, reliable rejection, camera transfer, or statistical superiority.

The evidence directory contains hashed copies of configs, histories, metrics and validation arrays, plus a hash for each local best checkpoint, but it does not copy the checkpoint bytes themselves. The checkpoints remain in the immutable local run directories; preserving those directories is required if later work needs exact weight recovery rather than receipt verification alone.
