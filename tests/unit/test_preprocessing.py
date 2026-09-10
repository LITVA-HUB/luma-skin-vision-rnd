import numpy as np
import pytest
from PIL import Image

from luma_skin_vision.preprocessing import decode_image, quality_gate


def test_decode_orientation_and_quality(tmp_path):
    arr = np.random.default_rng(1).integers(20, 230, (40, 80, 3), dtype=np.uint8)
    image = Image.fromarray(arr)
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    result = decode_image(path)
    assert result.shape == (80, 40, 3)
    assert quality_gate(np.zeros((64, 64, 3)), [0, 0, 64, 64])
    assert "face_too_small" in quality_gate(result, [0, 0, 20, 20])


def test_bad_icc_rejected(tmp_path):
    path = tmp_path / "bad.png"
    Image.new("RGB", (40, 40)).save(path, icc_profile=b"not-an-icc")
    with pytest.raises(ValueError, match="ICC"):
        decode_image(path)
