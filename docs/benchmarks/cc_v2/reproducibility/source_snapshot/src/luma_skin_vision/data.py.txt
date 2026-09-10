"""One JSONL record per image and measured facial region; no identity features."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

SPLITS = ("train", "validation", "calibration", "test")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["1.0"] = "1.0"
    data_kind: Literal["SYNTHETIC", "INSTRUMENT", "PHOTO_REFERENCE"]
    subject_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    session_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_mirrored: bool = Field(
        description="Whether stored post-EXIF pixels are mirrored; normalize before anatomical ROI"
    )
    device_manufacturer: str
    device_model: str = Field(min_length=1)
    camera_module: str
    front_or_rear: Literal["front", "rear", "synthetic"]
    os_version_if_relevant: str | None = None
    capture_app_version: str
    lighting_id: str = Field(min_length=1)
    lighting_type: str
    repeat_id: int = Field(ge=0)
    makeup_protocol: Literal["none", "present", "unknown", "synthetic"]
    distance: float = Field(gt=0, description="meters")
    pose_metadata_if_known: dict[str, float] | None = None
    exposure_metadata_if_available: dict[str, float] | None = None
    white_balance_metadata_if_available: dict[str, float] | None = None
    reference_region: Literal["left_cheek", "right_cheek"]
    ground_truth_L: float = Field(ge=0, le=100)
    ground_truth_a: float = Field(ge=-160, le=160)
    ground_truth_b: float = Field(ge=-160, le=160)
    reference_illuminant: Literal["D65"]
    reference_observer: Literal["2"]
    reference_instrument: str = Field(min_length=1)
    reference_measurement_id: str = Field(min_length=1)
    reference_geometry: str = Field(min_length=1)
    reference_mode: str = Field(min_length=1)
    reference_aperture_mm: float = Field(gt=0)
    reference_calibration_id: str = Field(min_length=1)
    reference_timestamp: str = Field(min_length=1)
    reference_repeats_lab: list[tuple[float, float, float]] = Field(min_length=2)
    face_bbox: tuple[float, float, float, float]
    split: Literal["train", "validation", "calibration", "test"]
    notes: str = ""

    @property
    def target(self):
        return np.array([self.ground_truth_L, self.ground_truth_a, self.ground_truth_b])

    @property
    def key(self):
        return f"{self.image_id}:{self.reference_region}"

    @model_validator(mode="after")
    def consistent(self):
        x, y, w, h = self.face_bbox
        if min(x, y) < 0 or min(w, h) <= 0:
            raise ValueError("invalid face bbox")
        reps = np.asarray(self.reference_repeats_lab)
        if np.any((reps[:, 0] < 0) | (reps[:, 0] > 100)):
            raise ValueError("invalid reference repeats")
        if not np.allclose(reps.mean(axis=0), self.target, atol=0.05, rtol=0):
            raise ValueError("ground truth must be mean of reference repeats within rounding 0.05")
        if self.data_kind == "INSTRUMENT" and (
            "synthetic" in self.reference_instrument.lower() or self.front_or_rear == "synthetic"
        ):
            raise ValueError("instrument provenance inconsistent")
        return self


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assign_splits(subject_ids, seed=42):
    ids = sorted(set(subject_ids))
    if len(ids) < 8:
        raise ValueError("At least 8 subjects required for four populated splits")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    sizes = [n - 3 * max(1, n // 5)] + [max(1, n // 5)] * 3
    return {
        s: split
        for split, group in zip(SPLITS, np.split(ids, np.cumsum(sizes)[:-1]), strict=True)
        for s in group
    }


def resolve_image(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("image path escapes dataset root")
    return path


def validate_records(manifest, *, verify_images=True):
    from PIL import Image

    manifest = Path(manifest)
    rows = [
        Record.model_validate_json(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("empty dataset")
    seen, subjects, images, hashes, references, captures, sessions = set(), {}, {}, {}, {}, {}, {}
    for r in rows:
        if r.key in seen:
            raise ValueError(f"duplicate image-region ID: {r.key}")
        seen.add(r.key)
        if r.subject_id in subjects and subjects[r.subject_id] != r.split:
            raise ValueError(f"subject leakage: {r.subject_id}")
        subjects[r.subject_id] = r.split
        fingerprint = (
            r.subject_id,
            r.session_id,
            r.image_path,
            r.device_model,
            r.camera_module,
            r.lighting_id,
            r.repeat_id,
            r.split,
            r.image_sha256,
            r.face_bbox,
            r.image_mirrored,
            r.device_manufacturer,
            r.front_or_rear,
            r.capture_app_version,
        )
        if r.image_id in images and images[r.image_id] != fingerprint:
            raise ValueError(f"inconsistent image ID: {r.image_id}")
        images[r.image_id] = fingerprint
        if r.image_sha256 in hashes and hashes[r.image_sha256] != (r.subject_id, r.split):
            raise ValueError("duplicate image content across subjects or splits")
        hashes[r.image_sha256] = (r.subject_id, r.split)
        capture_key = (
            r.subject_id,
            r.session_id,
            r.device_model,
            r.camera_module,
            r.lighting_id,
            r.repeat_id,
            r.reference_region,
        )
        if capture_key in captures:
            raise ValueError("duplicate capture metadata")
        captures[capture_key] = r.image_id
        ref = (
            r.subject_id,
            r.session_id,
            r.reference_region,
            tuple(r.target),
            r.reference_instrument,
            r.reference_geometry,
            r.reference_mode,
            tuple(r.reference_repeats_lab),
            r.reference_timestamp,
        )
        if (
            r.reference_measurement_id in references
            and references[r.reference_measurement_id] != ref
        ):
            raise ValueError("inconsistent reference measurement ID")
        references[r.reference_measurement_id] = ref
        session_key = (r.subject_id, r.session_id)
        protocol = (
            r.reference_instrument,
            r.reference_geometry,
            r.reference_mode,
            r.reference_aperture_mm,
            r.reference_calibration_id,
            r.data_kind,
        )
        if session_key in sessions and sessions[session_key] != protocol:
            raise ValueError("incompatible reference protocol in session")
        sessions[session_key] = protocol
        path = resolve_image(manifest.parent, r.image_path)
        if verify_images:
            if not path.is_file() or sha256(path) != r.image_sha256:
                raise ValueError(f"image missing or checksum mismatch: {r.image_id}")
            with Image.open(path) as im:
                x, y, w, h = r.face_bbox
                width, height = im.size
                if im.getexif().get(274, 1) in (5, 6, 7, 8):
                    width, height = height, width
                if x + w > width or y + h > height:
                    raise ValueError("face bbox outside image")
                im.verify()
    if len({r.data_kind for r in rows}) != 1:
        raise ValueError(
            "Do not mix synthetic, photo-reference and instrument targets in one dataset"
        )
    return rows


def dataset_identity(manifest, rows):
    split_payload = sorted((r.key, r.subject_id, r.split) for r in rows)
    return {
        "dataset_version": "1.0",
        "dataset_hash": sha256(manifest),
        "split_hash": hashlib.sha256(json.dumps(split_payload).encode()).hexdigest(),
        "data_kind": rows[0].data_kind,
    }
