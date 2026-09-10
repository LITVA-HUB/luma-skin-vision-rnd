# Deployment and resource boundary

Current artifact: local research engine; no production-supported operating domain. PyTorch MobileNetV3-small feature networks are initialized without external weights. CPU execution is supported; CUDA 12.8 wheels run on the detected RTX 4060. Inputs/configs start at 64 px for smoke. Real experiments must preregister 224–256 px candidates, microbatch 8, and measured memory limits; no accuracy statement transfers from 64 px toy patches.

Training extras contain torch/torchvision. Core schema/color/photometry/policy dependencies do not require them. Optional `face` contains OpenCV; optional `export` contains ONNX/ORT; optional `report` contains matplotlib. `load_run` of a learned checkpoint requires the training extra; a future ORT inference bundle must include preprocessing, residual prediction and calibration binding, not just the neural graph.

Current export uses explicitly pinned PyTorch 2.8 legacy TorchScript ONNX exporter, opset 17, FP32, dynamic batch and fixed resolution. Its deprecation warnings are recorded; numerical parity is tested for C/C+/proposed with nonzero output layers and on frozen synthetic test inputs. The unused ambiguity input is pruned for C and consumers inspect graph inputs. No claim is made that ONNX creates novelty or production readiness.

Benchmark reports identify backend/device, warmup, batch size, precision, measured latency distribution, parameter count and peak allocated VRAM. Current timings are model-only and exclude decode, face detection, ROI, ambiguity, error model and policy. Peak allocated CUDA memory is not total process/GPU usage. CPU RSS is not currently measured.

FP16 training is implemented behind a CUDA config and requires its own tests/accuracy comparison for real use. Production FP16 conversion, INT8, TensorRT and distillation are NOT STARTED; they are intentionally after positive real-data gates. Quantization must preserve color error and rejection behavior on frozen real data, and calibration may need refitting on an untouched calibration partition for the optimized model. No cloud GPU or paid service is required.
