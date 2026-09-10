import copy
import json

import pytest

from luma_skin_vision.data import Record, assign_splits, validate_records
from luma_skin_vision.synthetic import generate


def test_synthetic_valid_and_reproducible(tmp_path):
    manifest = generate(tmp_path / "a", subjects=20, seed=9)
    rows = validate_records(manifest, verify_images=True)
    other = generate(tmp_path / "b", subjects=20, seed=9)
    assert manifest.read_bytes() == other.read_bytes()
    assert len({r.subject_id for r in rows}) == 20
    assert {r.split for r in rows} == {"train", "validation", "calibration", "test"}
    assert all(r.data_kind == "SYNTHETIC" for r in rows)


def test_subject_leakage_and_duplicate(tmp_path):
    path = generate(tmp_path, subjects=20)
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    changed = copy.deepcopy(raw)
    changed[1]["split"] = "test" if changed[0]["split"] != "test" else "train"
    path.write_text("\n".join(json.dumps(r) for r in changed))
    with pytest.raises(ValueError, match="subject leakage"):
        validate_records(path)
    path.write_text("\n".join(json.dumps(r) for r in raw + [raw[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_records(path)


def test_reference_and_root_safety(tmp_path):
    path = generate(tmp_path, subjects=20)
    raw = json.loads(path.read_text().splitlines()[0])
    raw["reference_illuminant"] = "D50"
    with pytest.raises(ValueError):
        Record.model_validate(raw)
    raw["reference_illuminant"] = "D65"
    raw["image_path"] = "../outside.png"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="root"):
        validate_records(path)
    raw["image_path"] = "image.png"
    raw["ground_truth_L"] = None
    with pytest.raises(ValueError):
        Record.model_validate(raw)


def test_subject_split_order_independent():
    subjects = [f"s{i}" for i in range(20)]
    assert assign_splits(subjects, 17) == assign_splits(subjects[::-1], 17)
    assert assign_splits(subjects, 17) != assign_splits(subjects, 18)


def test_bbox_uses_post_exif_dimensions(tmp_path):
    from PIL import Image

    from luma_skin_vision.data import sha256

    path = generate(tmp_path, subjects=12)
    raw = json.loads(path.read_text().splitlines()[0])
    exif = Image.Exif()
    exif[274] = 6
    image = tmp_path / raw["image_path"]
    Image.new("RGB", (80, 40), (180, 120, 90)).save(image, exif=exif)
    raw["image_sha256"] = sha256(image)
    raw["face_bbox"] = [0, 0, 40, 80]
    path.write_text(json.dumps(raw))
    assert len(validate_records(path)) == 1
