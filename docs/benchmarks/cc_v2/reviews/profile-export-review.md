# CC v2 profile and portable export review — closed

Independent bounded local review, 2026-09-10, of `scripts/profile_cc_v2.py`, `tests/test_profile_cc_v2.py`, the called frozen inference/selector helpers, and `docs/benchmarks/cc_v2/profiles/profile.json`. Reviewed script SHA-256: `aeaae6674d453ca3f7a6877099d5d7518ebc3fe25baf3036e3bdf50ae294a4b8`; tests: `9d586446f79580079d5c0c5802dc4ddb8397a3f4a073e6a61037f64e626dc64c`.

**No blocker to running the guarded real-model export was found.** Its existing runtime parity check must pass before the resulting export is called verified. This review did not modify frozen source, dependency locks, selectors or trained artifacts. The reviewer used synthetic pixels only, read no real target labels/images and computed no target errors. The reported GPU profile was inspected and recalculated from saved samples, not rerun by this reviewer.

## Portable head and validity

The embedded Ridge calculation correctly applies the fitted `StandardScaler` mean/scale, Ridge coefficients/intercept, the shared `[0, log(181)]` clipping, `expm1 + 1e-8`, and positive source calibration scalar. The FP32 default casts these fitted constants without refitting. ONNX uses `exp - 1` because this export path lacks `expm1`; parity checks explicitly bound its numerical difference. The gate compares the cast head against sklearn and compares ONNX against PyTorch, requiring score/component errors below `1e-4` and exact validity/acceptance flags on the tested cases. The graph audit rejects DOUBLE tensors in the default FP32 export.

Validity is calculated from original pixels before sanitization: finite, nonnegative, and all channel means above EPS. Sanitization permits finite placeholder inference for malformed inputs, while the final acceptance flag always intersects the original validity mask. This matches the supported preprocessed-image contract; preprocessing clips thumbnails to `[0, 4]`. It does not establish quality or physical accuracy for an otherwise valid image. The exported boundary is FP32 `N x 3 x 128 x 128`; decoding and masking remain outside ONNX.

The source threshold uses the frozen calibration-score linear 80th percentile. It is an empirical acceptance threshold, not a promise of 80% target coverage. Floating-point parity and exact flags are measured on the tested inputs; inputs arbitrarily close to the threshold are not covered by a universal exact-decision guarantee.

The model/config/checkpoint/source are checked through `verify_run`; the final selector lock binds the selected head state. The selected head file is hash-checked before local joblib loading, and model/head/lock/helper hashes are checked again before measurement/export reporting. All recorded profile bindings currently match the local files. Raw PNGs selected for profiling are checked against source-manifest SHA-256 values. Only four source training rows are used; no fitting occurs here.

## Independent checks

- `.venv/Scripts/python -m pytest tests/test_profile_cc_v2.py -q`: **2 passed in 3.09 seconds**. Warnings concern the legacy exporter and traced assertions; the explicit exported validity operations and batch-1/batch-4 tests establish the tested runtime behavior.
- `.venv/Scripts/ruff check scripts/profile_cc_v2.py tests/test_profile_cc_v2.py`: **all checks passed**.
- Tests cover FP64 sklearn reference parity, FP32 ONNX dynamic batches 1 and 4, zero images, a missing channel, NaN, negative pixels and Inf, mandatory refusal, finite fallback outputs and FLOAT-only graph tensors.
- Additional independent probe loaded the actual hash-verified frozen SOG model and Ridge100 head, using eight synthetic arrays with varied exposure, masks and channel gains. FP32 embedded scores differed from sklearn by at most **`1.6030015972390288e-6` degrees**, below the `1e-4` gate. All acceptance flags and illuminant components matched exactly. No real image or label was involved.
- Independently counted **3,033,651 CNN parameters** and 85 Ridge coefficients. The parameter count is the CNN parameter count; scaler/Ridge/calibration buffers and serialized metadata are additional.

## Profile interpretation

All six sets of 100 saved samples reproduce their reported median, p95 and mean exactly. Each has 20 warmups and explicit GPU synchronization. Model-only input is already on the GPU. The thumbnail path includes H2D, model, relative features, D2H and the actual fitted CPU risk head. The PNG path additionally includes pixel decode, sensor-level normalization, target masking, resize/percentile normalization and cache-compatible float16 rounding; compressed PNG bytes are already in memory, so disk/ZIP I/O is excluded.

| Frozen model | Model median / p95 ms | Thumbnail-to-score median / p95 ms | PNG-bytes-to-score median / p95 ms |
|---|---:|---:|---:|
| SOG large, Ridge100 | 4.181 / 4.327 | 6.027 / 6.186 | 24.691 / 25.485 |
| Direct large, HGB3 | 4.083 / 4.284 | 7.378 / 7.482 | 30.307 / 36.476 |

The report's broad `FP32` label applies to the image/CNN feature path; the measured sklearn CPU head receives FP64 features and fitted constants. These timings are **PyTorch plus sklearn**, not ONNX or TensorRT measurements. FP32 ONNX export is not proof of TensorRT operator support, performance or deployability. The four repeatedly sampled source PNGs and one local session do not establish general latency distributions or a stable comparative speed advantage.

The approximately **28.12 MiB allocated / 36 MiB reserved** inference figures are absolute peaks from PyTorch's allocator during these measurements, including its starting allocations. The second model begins at **8.125 MiB allocated / 20 MiB reserved**. They are not incremental model memory, total process GPU memory, or total device usage. CUDA/driver/runtime/display allocations and CPU memory are outside these figures. The separate approximately 762.34/852 MiB training numbers explicitly include the source cache and must not be compared as equivalent inference measurements.

The final result remains an illuminant and empirical expected-error/acceptance component. The measured pipeline does not include correction/rendering, face analysis or physical facial color validation. No implementation change or refit is requested by this review; preserve the reporting qualifications above.

## Completed export evidence

After the review approved guarded execution, the implementing agent completed the real frozen-model export. The reviewer independently verified the saved report, identical documentation copy, every provenance binding, ONNX byte size and SHA-256. The export is **12,141,768 bytes**, SHA-256 `c619937943c34668d4e9970ee065218c6322e41a715e1f3c24f7b39062ecadbe`, with FLOAT as its only floating dtype and ONNX Runtime CPU provider.

The saved guarded check passed for source batches 1 and 4 and every malformed-input case. Its largest real-source ONNX/PyTorch score difference is **`4.291534423828125e-6` degrees**; flags match exactly. FP32/sklearn score difference on those four source inputs is at most **`7.325209390174336e-7` degrees**, with identical acceptance flags. These runtime measurements were produced by the implementing agent; the reviewer inspected and hash-verified their evidence rather than rerunning real-image inference. This closes the guarded export within the stated CPU-runtime and input-case scope.
