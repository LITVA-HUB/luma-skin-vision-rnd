# Frozen CCv2 inference profile

Measured on NVIDIA RTX 4060 and AMD Ryzen 9 7900X (12 cores, 24 logical processors), Windows, PyTorch 2.8.0+cu128, FP32 with TF32 disabled. Torch/OpenCV/BLAS use four CPU threads. Each path uses 20 warm-ups followed by 100 CUDA-synchronized wall-clock measurements, batch one.

| Seed-17 model and actual combined head | Model only, median / p95 ms | 128px input through model, relative features, CPU head, median / p95 ms | PNG bytes through score, median / p95 ms |
|---|---:|---:|---:|
| SoG large + Ridge100 | 4.181 / 4.327 | 6.027 / 6.187 | 24.691 / 25.485 |
| Direct large + HGB3 | 4.083 / 4.284 | 7.378 / 7.482 | 30.307 / 36.476 |

Both estimators have 3,033,651 parameters and 12,324,047-byte checkpoints. The model-only input is already on the GPU. The thumbnail path includes H2D transfer, model inference, relative features, D2H transfer, and the exact fitted sklearn risk head. The full path additionally decodes preloaded uint16 PNG bytes, subtracts the black level, applies saturation/target masks, area-resizes to 128px, normalizes exposure, and reproduces the cache's float16 rounding before FP32 inference. Disk and ZIP-container reads, full-resolution correction, rendering, and face analysis are excluded.

Four verified source images (00_0002, 00_0004, 00_0013, 00_0015), each 648×432, cycle through the measurements; both estimators use the same images. These are repeated latency observations, not 100 independent images. Direct-model full-path measurements drifted between roughly 24 and 37 ms. Sequential runs and different fitted head families preclude a strong claim of speed superiority.

Peak PyTorch allocator memory was approximately 28.12 MiB allocated / 36 MiB reserved for both runs, with no GPU dataset loaded. The first run started at zero allocated/reserved; the second retained 8.125/20 MiB of framework allocations from the earlier run, as recorded. These measurements exclude driver/CUDA-context allocations. Training reports separately record 762.344 MiB allocated / 852 MiB reserved, including a 233.452 MiB source cache.

Raw timings, source IDs/hashes, software versions, checkpoint/head/script/lock bindings and memory values are in `profile.json`. `profile_script_snapshot.py.txt` preserves the exact measurement script. This is an unoptimized eager-PyTorch pipeline measurement; it is not TensorRT performance or a skin-analysis/product latency claim.

## Portable component

The seed-17 SoG-large estimator and its frozen combined StandardScaler/Ridge100 head were exported together as `experiments/runs/ccv2_sog_large_g0_s17/export_v2/model.onnx` (12,141,768 bytes, SHA256 `c619937943c34668d4e9970ee065218c6322e41a715e1f3c24f7b39062ecadbe`). The ONNX graph contains FP32 floating tensors only, accepts dynamic batches of 3×128×128 preprocessed images, and outputs illuminant, expected reproduction error in degrees, validity, and acceptance at the frozen source-calibration 80th-percentile threshold (2.873881240803161°). Source calibration does not guarantee 80% coverage on new cameras.

ONNX Runtime 1.29.0 CPU versus PyTorch passed on four real source inputs at batch sizes one and four. Maximum real-input illuminant component difference was 8.94e-8; maximum score difference was 4.30e-6°. The FP32 embedded head differed from the original sklearn head by at most 7.33e-7°. All acceptance decisions matched. Synthetic zero, absent-channel, NaN, negative and infinite inputs produced finite fallback outputs with validity and acceptance both false. Original-input checks exist explicitly in the graph: traced PyTorch assertions are not relied upon for portable validation. The export uses Exp−1 for Expm1 and verifies the resulting numerical difference.

Exact vectors and provenance are in `export_report.json`, copied byte-for-byte from the export directory. Byte decoding and preprocessing remain outside this ONNX component. TensorRT compatibility, TensorRT latency and ONNX runtime latency have not been measured; the table above measures eager PyTorch plus the exact CPU sklearn head.
