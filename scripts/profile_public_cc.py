"""Unoptimized batch-one real input path, including CPU experts and error head."""

import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from luma_skin_vision.cc.benchmark import apply_risk, risk_features
from luma_skin_vision.cc.core import experts
from luma_skin_vision.cc.data import decode, sample
from luma_skin_vision.cc.model import CompactCC
from luma_skin_vision.experiment import write_json

root = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
cv2.setNumThreads(4)
rows = json.loads((root / "data/processed/cc128/cube_manifest.json").read_text())
rows = sorted([r for r in rows if r["subset"] == "train"], key=lambda r: r["id"])[:55]
archive = zipfile.ZipFile(root / "data/public/cube/SimpleCube++.zip")
pngs = [archive.read(f"SimpleCube++/train/PNG/{r['id']}.png") for r in rows]
results = {}
for method, mode in [("baseline", "context"), ("baseline", "combined"), ("proposed", "combined")]:
    folder = root / f"experiments/runs/cc_official_{method}_s17"
    saved = torch.load(folder / "model.pt", map_location="cuda", weights_only=True)
    model = CompactCC(saved["mixture"]).cuda().eval()
    model.load_state_dict(saved["state"])
    state = json.loads((folder / "risk_heads.json").read_text())[method + "_" + mode]
    state = {k: np.array(v) if isinstance(v, list) else v for k, v in state.items()}
    times = []
    with torch.inference_mode():
        for i, (raw, row) in enumerate(zip(pngs, rows)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            rgb = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)[..., ::-1]
            x = decode(rgb, white=row["white"])
            x[-250:, -175:] = 0
            thumbnail = sample(x)
            preprocess_end = time.perf_counter()
            # Standard C+ requires no expert features; placeholder branch does not affect its output.
            ex = experts(x) if mode == "combined" else np.ones((4, 3)) / np.sqrt(3)
            expert_end = time.perf_counter()
            pred, context = model(
                torch.tensor(thumbnail[None], device="cuda"),
                torch.tensor(ex[None], dtype=torch.float32, device="cuda"),
            )
            p, c = pred.cpu().numpy(), context.cpu().numpy()
            score = apply_risk(risk_features(c, p, ex[None], mode), state)
            corrected = x / p[0]
            assert np.isfinite(corrected).all() and np.isfinite(score).all()
            torch.cuda.synchronize()
            end = time.perf_counter()
            if i >= 5:
                times.append(
                    [
                        (preprocess_end - start) * 1000,
                        (expert_end - preprocess_end) * 1000,
                        (end - expert_end) * 1000,
                        (end - start) * 1000,
                    ]
                )
    a = np.array(times)
    results[method + "_" + mode] = {
        "samples": len(a),
        "warmup": 5,
        "preprocess_median_ms": float(np.median(a[:, 0])),
        "experts_median_ms": float(np.median(a[:, 1])),
        "gpu_transfer_model_risk_correction_median_ms": float(np.median(a[:, 2])),
        "full_path_median_ms": float(np.median(a[:, 3])),
        "full_path_p95_ms": float(np.percentile(a[:, 3], 95)),
    }
write_json(
    root / "docs/benchmarks/public_full_path_profile.json",
    {
        "hardware": "RTX4060 + Ryzen9 7900X, torch/OpenCV4 threads",
        "scope": "Unoptimized FP32, compressed PNG bytes already in RAM; decode/black mask/resize/expert bank/H2D/model/D2H/risk/full-resolution diagonal correction included. Disk read, UI and face analysis excluded.",
        "results": results,
    },
)
print(json.dumps(results, indent=2))
