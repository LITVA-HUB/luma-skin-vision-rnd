"""Deterministic toy patches, never evidence for real-world facial measurement."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from luma_skin_vision.color import linear_to_srgb, srgb_to_lab, srgb_to_linear
from luma_skin_vision.data import Record, assign_splits, sha256


def generate(root, subjects=20, seed=42, size=64):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = [f"syn_{i:03d}" for i in range(subjects)]
    splits = assign_splits(ids, seed)
    records = []
    for subject in ids:
        base = rng.uniform([0.25, 0.16, 0.10], [0.90, 0.78, 0.67])
        target = srgb_to_lab(base).tolist()
        for device in range(2):
            for light in range(2):
                for repeat in range(2):
                    image_id = f"{subject}_d{device}_l{light}_r{repeat}"
                    gain = np.array([[1.0, 1.0, 1.0], [1.08, 0.95, 0.90]])[device]
                    gain = gain * np.array([[1, 1, 1], [1.12, 0.94, 0.84]])[light]
                    shading = np.linspace(0.88, 1.08, size)[None, :, None]
                    rgb = np.broadcast_to(
                        srgb_to_linear(base) * gain * shading, (size, size, 3)
                    ).copy()
                    rgb += rng.normal(0, 0.002, rgb.shape)
                    encoded = np.clip(linear_to_srgb(rgb), 0, 1)
                    path = root / "images" / f"{image_id}.jpg"
                    Image.fromarray(np.uint8(np.round(encoded * 255))).save(path, quality=95)
                    for region in ("left_cheek", "right_cheek"):
                        row = Record(
                            data_kind="SYNTHETIC",
                            subject_id=subject,
                            session_id=subject + "_session",
                            image_id=image_id,
                            image_path=f"images/{image_id}.jpg",
                            image_sha256=sha256(path),
                            image_mirrored=False,
                            device_manufacturer="synthetic",
                            device_model=f"syn_camera_{device}",
                            camera_module="synthetic",
                            front_or_rear="synthetic",
                            capture_app_version="generator-v1",
                            lighting_id=f"syn_light_{light}",
                            lighting_type="synthetic channel gain",
                            repeat_id=repeat,
                            makeup_protocol="synthetic",
                            distance=0.5,
                            reference_region=region,
                            ground_truth_L=target[0],
                            ground_truth_a=target[1],
                            ground_truth_b=target[2],
                            reference_illuminant="D65",
                            reference_observer="2",
                            reference_instrument="synthetic analytical sRGB",
                            reference_measurement_id=subject + "_" + region,
                            reference_geometry="not physical",
                            reference_mode="analytical",
                            reference_aperture_mm=1,
                            reference_calibration_id="analytical-v1",
                            reference_timestamp="2026-09-10T00:00:00Z",
                            reference_repeats_lab=[tuple(target), tuple(target)],
                            face_bbox=(0, 0, size, size),
                            split=splits[subject],
                            notes="SYNTHETIC engineering smoke only; not a face or an instrument measurement",
                        )
                        records.append(row.model_dump())
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf-8"
    )
    return manifest
