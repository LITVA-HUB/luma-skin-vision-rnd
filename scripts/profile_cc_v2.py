"""Frozen CCv2 latency and portable inference; no fitting or accuracy optimization."""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import onnx
import onnxruntime as ort
import torch
from threadpoolctl import threadpool_limits

from luma_skin_vision.cc.data import decode, sample
from luma_skin_vision.cc.v2 import EPS, CompactResidualCC, risk_features_invariant
from luma_skin_vision.cc.v2_experiment import verify_run
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cc_v2_select", ROOT / "scripts/cc_v2_select.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class PortableCC(torch.nn.Module):
    """FP32 image/model/head by default; optional FP64 sklearn reference head.

    Validity is computed on original pixels. Sanitizing only enables finite
    fallback inference: invalid inputs always have accept80=False.
    """

    def __init__(self, model, payload, threshold, head_dtype=torch.float32):
        super().__init__()
        self.model = model
        if payload["block"] != "combined":
            raise ValueError("Portable component requires frozen combined head")
        scaler = payload["model"].named_steps["standardscaler"]
        ridge = payload["model"].named_steps["ridge"]
        if np.asarray(ridge.coef_).shape != (85,):
            raise ValueError("Expected 85-feature scalar Ridge")
        for name, value in {
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "coef": ridge.coef_,
            "intercept": ridge.intercept_,
            "cal_scale": payload["scale"],
            "threshold": threshold,
        }.items():
            self.register_buffer(name, torch.as_tensor(np.asarray(value), dtype=head_dtype))

    def forward(self, image):
        finite = torch.isfinite(image)
        valid = finite.flatten(1).all(1) & (image >= 0).flatten(1).all(1)
        valid = valid & (image.mean((-2, -1)) > EPS).all(1)
        safe = torch.where(finite, image, torch.zeros_like(image)).clamp_min(0)
        pred, context = self.model(safe)
        feature = risk_features_invariant(safe, pred, context)["combined"].to(self.mean.dtype)
        raw = ((feature - self.mean) / self.scale * self.coef).sum(1) + self.intercept
        clipped = raw.clamp(0, math.log(181))
        # ONNX has Exp but no Expm1. Exp-1 is algebraically equivalent;
        # explicit runtime parity checks bound the cancellation error.
        transformed = clipped.exp() - 1 if torch.onnx.is_in_onnx_export() else torch.expm1(clipped)
        score = (transformed + 1e-8) * self.cal_scale
        return pred, score, valid, valid & (score <= self.threshold)


def locked_model(run, data, lock_path):
    config, checkpoint = verify_run(run, data)
    lock, state = load(lock_path), load(run / "risk_v2/selection.json")
    if lock["cnn_selectors"].get(run.name) != sha256(run / "risk_v2/selection.json"):
        raise ValueError("Final selector lock mismatch")
    if (
        state["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
        or state["script_sha256"] != sha256(Path(selector.__file__))
        or lock["script_hashes"]["cc_v2_select.py"] != sha256(Path(selector.__file__))
    ):
        raise ValueError("Selector/checkpoint/helper binding mismatch")
    head_path = run / "risk_v2/combined.joblib"
    if sha256(head_path) != state["heads"]["combined"]["artifact_sha256"]:
        raise ValueError("Risk artifact changed")
    model = CompactResidualCC(config["mode"], config["backbone"]).eval()
    model.load_state_dict(
        torch.load(run / "model.pt", weights_only=True, map_location="cpu")["state"]
    )
    payload = joblib.load(head_path)  # Trusted local, hash-checked sklearn payload.
    paths = [
        run / name
        for name in (
            "config.json",
            "model.pt",
            "checkpoint_manifest.json",
            "training.json",
            "risk_v2/selection.json",
        )
    ]
    paths += [head_path, lock_path, Path(__file__), Path(selector.__file__)]
    bindings = {str(p.resolve()): sha256(p) for p in paths}
    return model, payload, state, {"bindings": bindings, "source_hash": config["source_hash"]}


def check_bindings(values):
    if any(sha256(Path(p)) != digest for p, digest in values.items()):
        raise ValueError("Artifact changed during measurement")


def preprocess(raw, row):
    rgb = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)[..., ::-1]
    x = decode(rgb, black=row["black"], white=row["white"])
    x[-250:, -175:] = 0
    # Match published cache precision before the FP32 inference boundary.
    return sample(x).astype(np.float16).astype(np.float32)


def source_inputs(data):
    rows = sorted(
        (r for r in load(data / "cube_manifest.json") if r["subset"] == "train"),
        key=lambda r: r["id"],
    )[:4]
    archive = ROOT / "data/public/cube/SimpleCube++.zip"
    with zipfile.ZipFile(archive) as z:
        pngs = [z.read(f"SimpleCube++/train/PNG/{row['id']}.png") for row in rows]
    for raw, row in zip(pngs, rows):
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Source PNG hash mismatch")
    images = torch.from_numpy(np.stack([preprocess(raw, row) for raw, row in zip(pngs, rows)]))
    return rows, pngs, images


def timing(call, warm=20, repeats=100):
    values = []
    for i in range(warm + repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call(i)
        torch.cuda.synchronize()
        if i >= warm:
            values.append((time.perf_counter() - start) * 1000)
    return {
        "warmup": warm,
        "samples": repeats,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "all_ms": values,
    }


def profile(run, data, lock_path, rows, pngs, images):
    model, payload, state, provenance = locked_model(run, data, lock_path)
    gc.collect()
    torch.cuda.empty_cache()
    initial = {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
    }
    model.cuda()
    single = images[:1].cuda()
    torch.cuda.reset_peak_memory_stats()

    def infer(x):
        x = x.cuda()
        pred, context = model(x)
        feature = risk_features_invariant(x, pred, context)
        pred.cpu().numpy()
        valid = feature["valid"].cpu().numpy()
        score = (
            selector.raw_predict(
                payload["model"], feature["combined"].cpu().numpy().astype(np.float64)
            )
            * payload["scale"]
        )
        return score, valid

    with torch.inference_mode(), threadpool_limits(limits=4):
        results = {
            "model_only": timing(lambda _: model(single)),
            "thumbnail_model_features_cpu_head": timing(lambda i: infer(images[i % 4 : i % 4 + 1])),
            "png_bytes_to_score": timing(
                lambda i: infer(torch.from_numpy(preprocess(pngs[i % 4], rows[i % 4])[None]))
            ),
        }
    training = load(run / "training.json")
    record = {
        **provenance,
        "run": run.name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_bytes": (run / "model.pt").stat().st_size,
        "risk_head_bytes": (run / "risk_v2/combined.joblib").stat().st_size,
        "head": state["heads"]["combined"]["selected"],
        "measurements": results,
        "initial_vram_no_dataset": initial,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "training_memory": {k: v for k, v in training.items() if "_mb" in k},
    }
    check_bindings(provenance["bindings"])
    model.cpu()
    single = single.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return record


def export_and_check(wrapper, images, path):
    if path.exists():
        raise FileExistsError(path)
    names = ["illuminant", "expected_reproduction_deg", "valid", "accept80"]
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (images[:1],),
            str(path),
            opset_version=18,
            dynamo=False,
            input_names=["image"],
            output_names=names,
            dynamic_axes={name: {0: "batch"} for name in ["image", *names]},
        )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    dtypes = {tensor.data_type for tensor in graph.graph.initializer}
    dtypes.update(
        value.type.tensor_type.elem_type
        for value in [*graph.graph.input, *graph.graph.output, *graph.graph.value_info]
    )
    dtypes.update(
        attribute.t.data_type
        for node in graph.graph.node
        for attribute in node.attribute
        if attribute.type == onnx.AttributeProto.TENSOR
    )
    float_dtypes = sorted(
        onnx.TensorProto.DataType.Name(dtype)
        for dtype in dtypes
        if dtype
        in (
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.BFLOAT16,
        )
    )
    if wrapper.mean.dtype == torch.float32 and "DOUBLE" in float_dtypes:
        raise ValueError("Unexpected FP64 tensor in FP32 portable graph")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    invalid = images.clone()
    invalid[0] = 0
    invalid[1, 1] = 0
    invalid[2, 0, 0, 0] = float("nan")
    invalid[3, 0, 0, 0] = -1
    infinite = images[:1].clone()
    infinite[0, 0, 0, 0] = float("inf")
    report = {
        "opset": 18,
        "runtime": ort.__version__,
        "providers": session.get_providers(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "cases": {},
        "floating_dtypes": float_dtypes,
        "head_dtype": str(wrapper.mean.dtype),
        "contract": "Nx3x128x128 preprocessed FP32 image; explicit finite/nonnegative/channel-mean validity; invalid always refused. Byte decode/preprocess outside ONNX.",
    }
    for name, x in {
        "real_batch1": images[:1],
        "real_batch4": images,
        "invalid_batch4": invalid,
        "infinite_batch1": infinite,
    }.items():
        with torch.inference_mode():
            expected = [value.numpy() for value in wrapper(x)]
        actual = session.run(None, {"image": x.numpy()})
        errors = {key: np.abs(actual[i] - expected[i]).tolist() for i, key in enumerate(names[:2])}
        flags = all(np.array_equal(actual[i], expected[i]) for i in (2, 3))
        if name.startswith(("invalid", "infinite")):
            flags = flags and not actual[2].any() and not actual[3].any()
        maximum = max(float(np.max(value)) for value in errors.values())
        passed = flags and all(np.isfinite(value).all() for value in actual[:2]) and maximum < 1e-4
        report["cases"][name] = {
            "absolute_error_vectors": errors,
            "max_absolute_error": maximum,
            "flags_exact": flags,
            "passed": passed,
            "torch_scores": expected[1].tolist(),
            "onnx_scores": actual[1].tolist(),
        }
    report["passed"] = all(case["passed"] for case in report["cases"].values())
    report["onnx_sha256"], report["onnx_bytes"] = sha256(path), path.stat().st_size
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profile", "export"))
    parser.add_argument("--head-precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/cc128")
    parser.add_argument(
        "--head-lock", type=Path, default=ROOT / "docs/benchmarks/cc_v2/final_head_lock.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/benchmarks/cc_v2/profiles/profile.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, pngs, images = source_inputs(args.data)
    if args.action == "profile":
        if args.output.exists():
            raise FileExistsError(args.output)
        report = {
            "hardware": {
                "gpu": torch.cuda.get_device_name(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "opencv": cv2.__version__,
                "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
            },
            "precision": "FP32, TF32 disabled, batch1, torch/OpenCV/BLAS4 threads",
            "scope": "Preloaded compressed PNG bytes; includes pixel decode, black/white and target mask, area resize/percentile normalization, cache-compatible float16 rounding, H2D/model/relative features/D2H/fitted CPU risk. Disk/ZIP I/O excluded; no correction/rendering/face analysis. Model-only input already on GPU. No GPU dataset loaded.",
            "source_rows": rows,
            "results": {},
        }
        for name in ("ccv2_sog_large_g0_s17", "ccv2_direct_large_g0_s17"):
            report["results"][name] = profile(
                ROOT / "experiments/runs" / name, args.data, args.head_lock, rows, pngs, images
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        print(json.dumps({n: r["measurements"] for n, r in report["results"].items()}), flush=True)
    else:
        run = ROOT / "experiments/runs/ccv2_sog_large_g0_s17"
        out = run / "export_v2"
        if out.exists():
            raise FileExistsError(out)
        model, payload, state, provenance = locked_model(run, args.data, args.head_lock)
        threshold = float(np.quantile(state["heads"]["combined"]["cal_scores"], 0.8))
        wrapper = PortableCC(
            model, payload, threshold, head_dtype=getattr(torch, args.head_precision)
        ).eval()
        with torch.inference_mode():
            pred, context = model(images)
            feature = (
                risk_features_invariant(images, pred, context)["combined"]
                .numpy()
                .astype(np.float64)
            )
            expected_score = selector.raw_predict(payload["model"], feature) * payload["scale"]
            _, score, valid, accept = wrapper(images)
        sklearn_reference = {
            "score_absolute_errors": np.abs(expected_score - score.numpy()).tolist(),
            "accept80_exact": bool(
                np.array_equal(accept.numpy(), valid.numpy() & (expected_score <= threshold))
            ),
        }
        sklearn_reference["passed"] = (
            max(sklearn_reference["score_absolute_errors"]) < 1e-4
            and sklearn_reference["accept80_exact"]
        )
        out.mkdir()
        write_json(out / "sklearn_reference.json", sklearn_reference)
        if not sklearn_reference["passed"]:
            raise RuntimeError(
                "Head cast failed sklearn parity; preserve evidence before choosing an explicit fallback"
            )
        report = export_and_check(wrapper, images, out / "model.onnx")
        report.update(
            provenance=provenance,
            source_rows=rows,
            threshold80=threshold,
            sklearn_reference=sklearn_reference,
            threshold_definition="NumPy linear quantile80 on268 frozen source calibration scores; no promised target coverage",
        )
        check_bindings(provenance["bindings"])
        write_json(out / "export_report.json", report)
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "max_errors": {k: v["max_absolute_error"] for k, v in report["cases"].items()},
                }
            ),
            flush=True,
        )
        if not report["passed"]:
            raise RuntimeError(
                "ONNX parity failed; preserve diagnostic artifact, do not claim verified export"
            )


if __name__ == "__main__":
    main()
