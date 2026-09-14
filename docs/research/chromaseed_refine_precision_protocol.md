# ChromaSeed-R storage precision follow-up

2026-09-13, registered after the primary FP32 study and before the precision evaluation. This is a post-hoc engineering follow-up on exposed exploratory rows, not new model validation. No optimizer updates, threshold adaptation or architecture selection.

Apply exactly two encodings to all 45 already selected final models:

1. FP16 storage of the complete learned parameter vector; keep input/target normalization, analytic anchor and exit threshold FP32.
2. Symmetric INT8 weights per output channel in every dense layer, scale=max(abs(weight))/127, round-to-nearest with clipping [-127,127]. Zero channels use scale 1. Keep layer biases, scales, normalization, anchor and exit threshold FP32.

Decode once to FP32 for the existing independently audited NumPy inference path. Thus this measures storage/download size and prediction drift, not FP16/INT8 arithmetic or a kernel speedup. Save encoded files and compare against the very same FP32 model in NumPy to isolate serialization error from framework differences. Retain the original exit thresholds and report any changed exit steps or patch counts.

Measure actual numerical/ZIP bytes, person-balanced native DeltaE00, change in error, maximum/p95 DeltaE00 between encoded and original predictions, and changed-exit fraction on every historical outer row in all three protocols. A provisional engineering guard is maximum prediction drift ≤0.1 DeltaE00 and person-mean error increase ≤0.02 in each individual model. This tolerance is not a validated cosmetics requirement. Report failures without weakening the guard or selecting a favorable seed/protocol.

The original trained FP32 files and source lock remain immutable. Encoding/audit source hashes and encoded model hashes go to a distinct ignored run directory. No participant images or identifiers are exported.
