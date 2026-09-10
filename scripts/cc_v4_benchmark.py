"""Measured batch-one cached/refined inference; no export or test-set access."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v4_model import CorrectionEvidenceNet

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", default=100, type=int)
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.exists() or args.repeats < 20:
        raise ValueError("Fresh output and at least20 measured repetitions required")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    training_result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if not config["is_frozen_primary"] or training_result["status"] != "complete":
        raise ValueError("Definitive timing requires a completed frozen primary run")
    receipt = json.loads((run / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, expected in receipt["sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"Run artifact changed: {name}")
    if sha256(ROOT / "scripts/cc_v4_model.py") != config["scripts_sha256"]["cc_v4_model.py"]:
        raise ValueError("Model implementation differs from the trained snapshot")
    if sha256(ROOT / "scripts/cc_v2_statistics.py") != config["scripts_sha256"]["cc_v2_statistics.py"]:
        raise ValueError("Source row decoder differs from the trained snapshot")
    data = Path(config["arguments"]["data"]).resolve()
    for name, expected in config["data_hashes"].items():
        if sha256(data / name) != expected:
            raise ValueError("Source cache changed")
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    selected = np.array([i for i, row in enumerate(rows) if row["id"] in config["validation_ids"][:4]])
    if len(selected) != 4:
        raise ValueError("Four recorded source-validation rows required")
    # Only these image rows are decoded, no GT key and no other camera/domain.
    images = read_npz_rows(data / "cube.npz", "images", selected, 2234).astype(np.float32)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = CorrectionEvidenceNet(mode=config["arguments"]["mode"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    x = torch.from_numpy(images).cuda()
    torch.cuda.synchronize()
    result = {
        "schema": "cc-v4-batch1-inference-1", "gpu": torch.cuda.get_device_name(),
        "mode": config["arguments"]["mode"], "run": str(run), "is_frozen_primary": config["is_frozen_primary"],
        "training_status": training_result["status"], "config_sha256": sha256(run / "config.json"),
        "run_manifest_sha256": sha256(run / "artifact_manifest.json"),
        "decoder_sha256": config["scripts_sha256"]["cc_v2_statistics.py"],
        "torch": torch.__version__, "checkpoint_sha256": sha256(run / "best.pt"),
        "benchmark_script_sha256": sha256(Path(__file__)), "model_sha256": config["scripts_sha256"]["cc_v4_model.py"],
        "input_ids": [rows[i]["id"] for i in selected], "input_shape": list(images.shape[1:]),
        "input_count": 4, "repeats_per_stage": args.repeats, "warmup_per_stage": 20,
        "units": "milliseconds; synchronized host wall time, FP32 batch1",
        "scope": "Device-resident128x128RGB -> encoder once -> all candidate queries and selected risk; excludes image decoding/disk/host-device upload/correction rendering",
        "memory": "PyTorch allocator; excludes CUDA driver/context. Baseline includes model and four128RGB input tensors; incremental peak is above that baseline.",
        "parameters": sum(p.numel() for p in model.parameters()), "stages": {},
    }
    with torch.inference_mode():
        for steps in (1, 2, 4):
            for i in range(20):
                cache = model.encode(x[i % 4:i % 4 + 1])
                output = model.select(cache, steps=steps)
            del cache, output
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            samples, queries = [], set()
            for i in range(args.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                cache = model.encode(x[i % 4:i % 4 + 1])
                output = model.select(cache, steps=steps)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
                queries.add(output["query_count"])
                if not output["valid"].all() or not torch.isfinite(output["pred"]).all():
                    raise ValueError("Invalid benchmark output")
                del cache, output
            expected_queries = {1: 25, 2: 51, 4: 103}[steps]
            if queries != {expected_queries}:
                raise ValueError("Observed inference budget differs from protocol")
            peak = torch.cuda.max_memory_allocated()
            result["stages"][str(steps)] = {
                "median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
                "min_ms": min(samples), "samples_ms": samples, "observed_queries": sorted(queries),
                "baseline_allocated_mib": baseline / 2**20, "peak_allocated_mib": peak / 2**20,
                "incremental_peak_mib": (peak - baseline) / 2**20,
            }
    write_json(out, result)
    print(json.dumps({"out": str(out), "stages": {k: {a: b for a, b in v.items() if a != "samples_ms"} for k, v in result["stages"].items()}}))


if __name__ == "__main__":
    main()
