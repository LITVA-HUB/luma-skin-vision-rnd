"""FP32 export verification only; production optimization awaits real evidence."""

import time
from pathlib import Path

import numpy as np

from luma_skin_vision.color import delta_e00
from luma_skin_vision.experiment import write_json


def ort_inputs(session, images, ambiguity):
    inputs = {
        "image": np.asarray(images, dtype=np.float32),
        "ambiguity": np.asarray(ambiguity, dtype=np.float32),
    }
    return {i.name: inputs[i.name] for i in session.get_inputs()}


def export_model(model, path, images, ambiguity):
    import onnx
    import onnxruntime as ort
    import torch

    path = Path(path)
    model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (images[:1], ambiguity[:1]),
            path,
            input_names=["image", "ambiguity"],
            output_names=["lab"],
            opset_version=17,
            dynamo=False,
            dynamic_axes={"image": {0: "batch"}, "ambiguity": {0: "batch"}, "lab": {0: "batch"}},
        )
        reference = model(images, ambiguity).numpy()
    onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    output = session.run(None, ort_inputs(session, images.numpy(), ambiguity.numpy()))[0]
    difference = float(np.abs(output - reference).max())
    de = float(delta_e00(reference, output).max())
    if difference > 0.01 or de > 0.01:
        raise ValueError(f"ONNX equivalence failed: max Lab={difference}, deltaE={de}")
    return {
        "max_abs_lab_difference": difference,
        "max_delta_e00_difference": de,
        "onnx_bytes": path.stat().st_size,
        "samples": len(images),
        "precision": "FP32",
        "provider": "CPUExecutionProvider",
        "inputs": [i.name for i in session.get_inputs()],
        "opset": 17,
    }


def export_run(run):
    import torch

    from luma_skin_vision.data import sha256, validate_records
    from luma_skin_vision.training import LEARNED, load_run, prepare

    run = Path(run)
    model, meta = load_run(run)
    if meta["config"]["method"] not in LEARNED:
        raise ValueError("ONNX export requires a learned model")
    if sha256(meta["config"]["dataset"]) != meta["dataset_hash"]:
        raise ValueError("dataset changed since training")
    rows = [r for r in validate_records(meta["config"]["dataset"]) if r.split == "test"]
    data = prepare(meta["config"]["dataset"], rows, meta["config"])
    report = export_model(
        model,
        run / "model.onnx",
        torch.from_numpy(data["images"]).float(),
        torch.from_numpy(data["aux"]).float(),
    )
    report.update(
        data_kind=meta["data_kind"],
        dataset_hash=meta["dataset_hash"],
        checkpoint_sha256=meta["checkpoint_sha256"],
        onnx_sha256=sha256(run / "model.onnx"),
        scope="engineering numerical equivalence on frozen test inputs; not production readiness",
    )
    write_json(run / "export_report.json", report)
    return report


def benchmark_run(run, *, iterations=50, device="cpu", onnx=False):
    from luma_skin_vision.environment import inspect_environment
    from luma_skin_vision.training import LEARNED, load_run

    if iterations < 5:
        raise ValueError("at least five timed iterations required")
    run = Path(run)
    model, meta = load_run(run)
    if meta["config"]["method"] not in LEARNED:
        raise ValueError("model-only benchmark requires learned model")
    size = meta["config"]["resolution"]
    rng = np.random.default_rng(1)
    images = rng.uniform(size=(1, 3, size, size)).astype(np.float32)
    aux = np.zeros((1, 12), dtype=np.float32)
    peak = None
    if onnx:
        import onnxruntime as ort

        if device != "cpu":
            raise ValueError("This ORT package benchmark implements CPU provider only")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(run / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        values = ort_inputs(session, images, aux)

        def invoke():
            return session.run(None, values)

        def sync():
            return None

        backend = "onnxruntime CPUExecutionProvider"
    else:
        import torch

        if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("benchmark device unavailable")
        torch.set_num_threads(4)
        model.to(device).eval()
        x, a = torch.from_numpy(images).to(device), torch.from_numpy(aux).to(device)

        def invoke():
            with torch.inference_mode():
                return model(x, a)

        sync = torch.cuda.synchronize if device == "cuda" else lambda: None
        backend = "pytorch"
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        invoke()
    sync()
    timings = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        invoke()
        sync()
        timings.append(1000 * (time.perf_counter() - start))
    if not onnx and device == "cuda":
        peak = torch.cuda.max_memory_allocated()
    report = {
        "data_kind": "SYNTHETIC",
        "scope": "model-only batch-1; excludes decode, face, ROI, error model and policy",
        "backend": backend,
        "device": device,
        "precision": "FP32",
        "resolution": size,
        "batch_size": 1,
        "warmup": 10,
        "iterations": iterations,
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.quantile(timings, 0.95)),
        "peak_vram_bytes": peak,
        "parameters": meta["parameters"],
        "hardware": inspect_environment(),
    }
    write_json(run / f"benchmark_{'onnx' if onnx else 'torch'}_{device}.json", report)
    return report
