# Measured deployment engineering smoke — 2026-09-10

**SYNTHETIC, FP32, 64×64 crops, batch 1. Model-only timing: excludes decoding, face detection, ROI selection, ambiguity computation, residual model and policy. These are not production/phone latency claims.** Ten warmup calls and 50 timed iterations per backend, four CPU threads, GPU synchronized around timing. The machine was an ordinary active workstation, not a controlled performance lab.

| Model | Parameters | ONNX bytes | CPU PyTorch median ms | RTX 4060 median ms | CPU ORT median ms | Export max ΔE00 discrepancy |
|---|---:|---:|---:|---:|---:|---:|
| baseline_c | 1001251 | 4014595 | 2.4141 | 3.4848 | 0.3268 | 4.6708548e-07 |
| baseline_c_plus | 1002596 | 4022684 | 2.6719 | 3.2017 | 0.3334 | 3.4671957e-06 |
| proposed_v1 | 1003364 | 4025304 | 2.7541 | 3.1960 | 0.4190 | 3.602762e-06 |

| Model | Complete training + OOF seconds | Peak CUDA allocated bytes | Training device |
|---|---:|---:|---|
| baseline_c | 8.920 | 43435008 | cuda |
| baseline_c_plus | 8.917 | 43459584 | cuda |
| proposed_v1 | 8.831 | 43471872 | cuda |

Export equivalence uses all frozen synthetic test-region inputs; the ONNX artifact is bound to the originating checkpoint and dataset by SHA256. GPU VRAM denotes peak allocated tensors across training/OOF, not all GPU memory in use.

FP16 production, INT8, TensorRT, teacher/student: NOT STARTED pending useful real-data evidence. Parameter count and file size are measured artifact properties, not evidence of novelty. Raw p95 timings/environment/hashes: [synthetic_smoke_evidence.json](synthetic_smoke_evidence.json).
