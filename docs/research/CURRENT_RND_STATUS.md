# Current R&D status — camera-transfer CC v2, 2026-09-10

**Measured improvement against the strongest matched direct CNN: reproduction error at 80% acceptance on 384 fresh unseen-camera images falls from 5.558° to 3.805° (31.5%). No universal best-method, patent-novelty or facial skin-color claim is established.**

The proprietary instrument-paired facial dataset remains unavailable under the user's hard constraint. Public-data R&D continues. Historical synthetic state `2685bf0` / `milestone/synthetic-only-2026-09-10` and public V1 state `7637d6d` / `milestone/public-cc-feasibility-2026-09-10` remain frozen. Their negative results are preserved in [synthetic status](SYNTHETIC_MILESTONE_STATUS.md), [V1 status](PUBLIC_V1_MILESTONE_STATUS.md) and [V1 report](../benchmarks/public_benchmark_report.md).

## Implemented and measured

- Train only on SimpleCube++ real illuminant ground truth, CC BY 4.0. Separate capture-date groups for estimator fitting, model validation, error fitting and calibration. No imported pretrained weights.
- Fresh evaluation-only INTEL-TAU subset: 128 images each from Canon 5DSR, Nikon D810 and Sony IMX135_BLCCSC. Original CC BY-SA 4.0 verified directly. Deterministic IDs, 4.434 GB transferred, all 768 image/GT files hash-checked. Pinned mirror consistency is verified; byte identity to original full archives is not. This is a custom subset, not the full published benchmark protocol.
- A known anchor-normalized residual construction with a source-fitted relative-feature error head. Seventeen CNN trainings, matched 0.964M/3.034M variants, three final seeds, source-only selector and statistics-regression screens, preserved failed variants, all final choices locked before new-camera errors.
- Strong matched 3.034M direct C+ and Proposed receive the same 128×128 input, data, scratch initialization, 120 epochs, optimizer/augmentation budget and selector search. Classical Gray World, Max RGB, Shades of Gray and Gray Edge, compact learned statistics controls and preserved V1 methods are locally evaluated.
- Recovery/reproduction errors, all fixed coverage levels, complete curves, tails, paired bootstrap, per-camera errors and realized coverage at frozen source thresholds. Independent recomputation of 594 records agrees to numerical precision.

Fresh full mean: Proposed 4.719° versus C+ 5.662°. Fresh risk80: 3.805° versus 5.558°, paired difference −1.753° with proxy-cluster 95% interval [−2.132,−1.336]. Same-estimator context-only versus combined-risk ablation improves 4.266°→3.805°. Cheap GW+ridge achieves 4.558° full mean and 3.977° risk80; its risk80 difference from Proposed is inconclusive. A one-seed GW residual diagnostic has lower full mean, 4.229°.

**Negative findings:** source risk80 worsens 1.626°→2.276°; Canon risk80 worsens 4.042°→4.683°. Nikon and Sony improve versus C+, but there is no camera-uniform dominance. A source-calibrated nominal80% threshold accepts only47.66% of new images. Accepted catastrophic errors remain. Source and target differ in scenes/dataset as well as camera; physical scene independence is not established.

## Compute and export

Representative seed17: 3,033,651 parameters, 12.324 MB checkpoint, 12.142 MB FP32 ONNX. RTX4060 batch1 median4.181 ms model-only;24.691 ms PNG bytes→score at648×432 source image size, excluding disk and correction rendering. Peak PyTorch allocated training762.34 MiB includes233.45 MiB cached source data; inference28.12 MiB, excluding driver/context memory. ONNX CPU parity passed, maximum score difference4.30e-6°, all tested invalid inputs refused. TensorRT and quantized accuracy/latency remain unmeasured.

## What remains unvalidated and next decision

Prioritize compact single-image unseen-camera transfer (candidate C), retaining relative-context risk estimation (B) and the limited reproduction-risk proxy (A). The equivariant wrapper is known prior art, not a new invention. Keep cheap GW+ridge and GW-residual controls. Next: independent multi-camera source validation, stronger published compact-method reproduction and a new locked test set; investigate Canon failure without tuning this now-observed subset.

Physical facial Lab/ΔE00, skin-specific reliability, arbitrary smartphone ISP robustness, cosmetics outcomes, production calibration guarantees and patentability remain **NOT VALIDATED**. Skin-specific colorimetric validation remains future work requiring a facial dataset with appropriate reference measurements.

Full evidence: [V2 report](../benchmarks/cc_v2_report.md), [reproduction](../benchmarks/cc_v2/REPRODUCE.md), [prior art](cc_v2_prior_art.md), [licenses](../data/public_dataset_inventory.md), [Skolkovo addendum](../skolkovo/cc_v2_evidence_addendum.md).
