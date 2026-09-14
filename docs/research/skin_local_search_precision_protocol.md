# Post-training storage precision protocol

Status: PREPARE IMPLEMENTED. Outer evaluation is not run until the user explicitly approves it.

## Question and evidence boundary

This is a storage-only follow-up to `skin_local_search_v1`. It asks how much the 33 already
frozen final model payloads can be reduced before their Lab predictions materially depart from
the canonical FP32 payload. It does not retrain a model, choose a method, validate phone-face
accuracy, or make a native INT8 latency claim.

The prepare phase opens only these primary model artifacts under the three protocol `final`
directories: seed 17 for ridge and KRR, and seeds 17, 29 and 43 for random RBF, guided RBF and
MLP. The inventory is fixed at 11 models per protocol and 33 total. Files ending `_outer.npz`,
row-level predictions, targets, validation, calibration and test data are not opened. Prepare
writes gitignored artifacts under `experiments/runs/skin_local_search_v1/precision`, then freezes
the source and artifact SHA256 values in `manifest.json` and its separate `manifest.sha256`
receipt. A later source, artifact or manifest change is rejected.

## Fixed storage formats

All formats preserve the method identifier. Numeric payload counts include preprocessing,
data-derived centers, coefficients, biases, widths and every quantization scale.

1. `fp32_reference`: every numeric source array remains FP32. This is an independently serialized
   reference for the storage comparison.
2. `fp16`: every numeric array, including input and target normalization, is stored in FP16.
   Nonfinite FP16 conversion is rejected. At inference all arrays are materialized as FP32 before
   the existing NumPy predictor runs.
3. `int8`: `x_mean`, `x_std`, `y_mean` and `y_std` remain FP32. Other numeric arrays use symmetric
   signed INT8 with round-to-nearest and range `[-127, 127]`. A zero-valued quantization group gets
   scale 1, so stored scales are always finite and strictly positive. `beta` uses one scale per
   output column; centers use one scale per input feature; each MLP weight matrix uses one scale
   per output row. Each bias vector and the whole width array use one scale. Inference dequantizes
   these arrays to FP32 before calling the existing predictor.

The manifest records actual numeric scalars and bytes by dtype, scale-scalar count, NPZ archive
bytes, and hashes. Archive size includes container overhead and is distinct from numerical payload
bytes. INT8 and FP16 are storage representations followed by FP32 NumPy inference; they are not
native low-precision kernels, so this experiment supports no speed or energy claim.

## Frozen outer evaluation

Evaluation is a separate command and remains unexecuted during prepare. When authorized, it must
use only the hash-locked original TRAIN cache and the three already frozen outer role files from
`skin_local_search_v1`. It evaluates all three formats for every one of the 33 source models, for
99 records total. No format is selected, rejected or tuned on an outer role.

For each record, report the original protocol's person-balanced mean CIEDE2000 against native Lab,
the remaining outer aggregates, stored numerical payload, and prediction deviation from that
model's FP32 reference. Deviation contains maximum absolute Lab-component error, mean and maximum
Euclidean Lab distance, and image- and person-balanced CIEDE2000. The three outer protocols overlap
within original TRAIN and include historically exposed people, so their results remain exploratory.

## Commands

Prepare, which does not open the data cache:

```powershell
.venv/Scripts/python scripts/skin_local_search_precision.py --stage prepare
```

Evaluate only after explicit approval, with the same absolute licensed TRAIN cache used by the
parent experiment:

```powershell
.venv/Scripts/python scripts/skin_local_search_precision.py --stage evaluate --cache <absolute-original-TRAIN-cache>
```
