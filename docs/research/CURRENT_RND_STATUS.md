# Current R&D status — V5 paired training in progress, 2026-09-11

Execution update: seeds17/29 completed all six arms each. All24 best/final
prediction records passed independent scoring and CPU checkpoint replay. New
selected-action/gradient training has no consistent advantage so far; seed43
continues in the same serial session. Phone
coverage is now an explicit priority: a licensed Samsung S21 Plus/Oppo Find X5
Pro subset is downloading, and a small CC0 iPhone SE2/XS Max repeatability archive
is acquired. Phone loading/reference extraction and classical diagnostics passed
on six TRAIN captures; no held-out phone accuracy has been measured. [Current handles, interim
numbers and next steps](cc_v5_execution_status.md),
[phone-data rights and protocol](../data/smartphone_benchmark_plan.md).

V5 is now testing selected-action error supervision and physical cost derivatives
against point-only, posterior and generic action controls. Six arms per seed,
seeds17/29/43, 120 epochs. All arms clone complete20-epoch warmup state;
strict CUDA warmup replay passed bitwise. Primary execution was launched in
session74620 from committed training code2b23f90. This is an execution receipt,
not evidence that all runs completed. Read run result.json/session state before
assuming completion or restarting. [Protocol](cc_v5_protocol.md),
[preflight](../benchmarks/cc_v5_preflight.md),
[source binding](../benchmarks/cc_v5/launch_receipt.json).

The full suite passed190 tests before primary launch. Two additional independent
scorer tests subsequently passed. V5 primary accuracy, calibration and camera
generalization remain pending. The separate scorer verifies run hashes and
recomputes errors from real validation GT, retaining best and final checkpoints.
The strongest completed positive milestone remains V2; the latest completed
architecture screen remains the negative V4 result below.

## Preserved V4 screen

**Latest V4 architecture does not beat its controls.** Three 3.097M-parameter, 120-epoch runs give step-2 mean reproduction error 2.389° for physical evidence transport, 2.317° for generic action feedback, and 2.181° for the action-independent posterior. Raw risk80 is 2.000° / 2.009° / 1.895°, respectively. These 119 validation images were reused for development, with one seed; nondeterministic CUDA histories diverged even during the common warmup. This is a negative screen for the proposed mechanism, not a stable causal comparison or an independent benchmark. [V4 report](../benchmarks/cc_v4_report.md), [architecture/training synthesis](cc_v4_universal_method.md).

The user's repeated-refinement idea was implemented and measured at 1/2/4 stages. Posterior mean 2.186°→2.181°→2.174° is only a small change; transport 2.357°→2.389°→2.371° worsens relative to its starting estimate. Predicted risk decreases while some true errors increase. Proposed step-2 batch-1 median latency was 6.104–6.203 ms across three sessions on RTX 4060, from device-resident 128×128 RGB through the cached encoder and all queries. The model has 3,097,189 parameters and a 12,599,089-byte checkpoint. Training peak allocation was 765.46 MiB including 233.45 MiB of source cache; inference peak was 29.12 MiB. Other mode/stage timings vary more on the desktop; all receipts are retained. No V4 export optimization was performed.

Modern leading-lab training and close photometric sources were reviewed; no external pretrained weights were adopted. Next priority shifts from adding refinement stages to improving selected-action error supervision with paired initial checkpoints and stronger scene context. New-camera, source-test and calibration roles remain untouched by V4. [Decision](cc_v4_next_decision.md). The repository passed 185 tests; independent metric recomputation and CPU checkpoint replay passed. The overall R&D goal remains active.

## Preserved V3 architecture screen

**Latest experiment: the new full color-frame graph loses on real development validation.** Three matched1.216M/120epoch runs give mean reproduction4.281° full-frame Proposed versus2.547° direct RGB and2.942° diagonal. At80% raw-posterior acceptance, full-frame gives4.155° versus2.304° direct. These119 validation images were reused for development; this is not a fresh test result. The negative design is preserved and a relaxed graph with explicit global color state is the next unverified hypothesis. [V3 report](../benchmarks/cc_v3_report.md), [revision decision](cc_v3_revision_decision.md).

The384 newly acquired INTEL-TAU images have not yet entered training or error evaluation. Identity auditing found67 rows sharing old reference-file hashes, despite zero image duplicates; fold/group policy remains to be locked. A source-grounded FFCC numerical control is implemented but not yet trained. The broader R&D goal remains active.

## Preserved V2 measured milestone

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
