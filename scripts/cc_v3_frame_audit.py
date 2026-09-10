"""Source-only conditioning audit for the proposed full color frame; no target evaluation."""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from cc_v2_statistics import read_npz_rows

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def main():
    data = Path("data/processed/cc128")
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    selected = np.array([i for i, r in enumerate(rows) if r["subset"] in ("train", "val")])
    x = read_npz_rows(data / "cube.npz", "images", selected, expected_rows=2234).astype(np.float64)
    gt = read_npz_rows(data / "cube.npz", "gt", selected, expected_rows=2234).astype(np.float64)
    if x.shape[2:] != (128, 128):
        raise ValueError("Expected original128source thumbnails")
    patch = x.reshape(-1, 3, 4, 32, 4, 32).mean(axis=(3, 5)).reshape(-1, 3, 16)
    triples = np.array(list(itertools.combinations(range(16), 3)))
    candidates = patch[:, :, triples].transpose(0, 2, 1, 3)
    volumes = np.abs(np.linalg.det(candidates))
    best = volumes.argmax(axis=1)
    frame = candidates[np.arange(len(x)), best]
    singular = np.linalg.svd(frame, compute_uv=False)
    cond = singular[:, 0] / np.maximum(singular[:, -1], 1e-300)
    sorted_volume = np.sort(volumes, axis=1)
    gap = (sorted_volume[:, -1] - sorted_volume[:, -2]) / np.maximum(sorted_volume[:, -1], 1e-300)
    mixing = np.array([[1.05, 0.10, 0.03], [0.04, 0.90, 0.07], [0.02, 0.08, 1.10]])
    transformed = mixing @ candidates
    transformed_best = np.abs(np.linalg.det(transformed)).argmax(axis=1)
    usable = cond < 1e6
    inv_frame_gt = np.linalg.solve(frame[usable], gt[usable, :, None]).squeeze(-1)
    inv_frame_gw = np.linalg.solve(frame[usable], x[usable].mean(axis=(2, 3))[:, :, None]).squeeze(
        -1
    )
    cosine = np.sum(inv_frame_gt * inv_frame_gw, axis=1) / (
        np.linalg.norm(inv_frame_gt, axis=1) * np.linalg.norm(inv_frame_gw, axis=1)
    )
    canonical_angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    repro = gt / np.maximum(x.mean(axis=(2, 3)), 1e-12)
    repro_angle = np.degrees(
        np.arccos(np.clip(repro.sum(axis=1) / (np.sqrt(3) * np.linalg.norm(repro, axis=1)), -1, 1))
    )
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Only source1126train+119val rows decoded; test/risk/cal/INTEL excluded. Color mixing is a constructed numerical probe, not new real camera ground truth.",
        "images": len(x),
        "data_hashes": {name: sha256(data / name) for name in ("cube.npz", "cube_manifest.json")},
        "script_sha256": sha256(Path(__file__)),
        "row_ids": [rows[i]["id"] for i in selected],
        "condition_quantiles_0_25_50_75_90_95_99_100": np.percentile(
            cond, [0, 25, 50, 75, 90, 95, 99, 100]
        ).tolist(),
        "condition_exceeds": {str(t): int((cond > t).sum()) for t in (100, 1000, 1e4, 1e5, 1e6)},
        "near_tie_fraction_gap_under_1e5": float((gap < 1e-5).mean()),
        "positive_mixing_matrix": mixing.tolist(),
        "changed_frame_triples_under_mixing": int((best != transformed_best).sum()),
        "canonical_gt_vs_gw_angle_quantiles_0_25_50_75_90_95_99_100": np.percentile(
            canonical_angle, [0, 25, 50, 75, 90, 95, 99, 100]
        ).tolist(),
        "ordinary_gw_reproduction_mean": float(repro_angle.mean()),
        "inference": "Large canonical angular separation despite small camera reproduction error would warn against letting canonical NLL dominate physical task loss. Frame rank is not a sufficient reliability certificate.",
    }
    output = Path("docs/benchmarks/cc_v3/source_frame_conditioning.json")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("row_ids", "data_hashes")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
