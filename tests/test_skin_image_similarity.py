import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from skin_image_similarity import candidates, describe, similarity  # noqa: E402


def picture(seed=7):
    rng = np.random.default_rng(seed)
    image = gaussian_filter(rng.normal(size=(192, 192, 3)), sigma=(4, 4, 0))
    return np.clip(image / image.std() * 35 + 128, 0, 255).astype(np.uint8)


def test_same_pixels_and_horizontal_reflection_match_without_a_neural_model():
    rgb = picture()
    thumb, hashes, std = describe(rgb)
    flipped, flipped_hashes, _ = describe(rgb[:, ::-1])
    assert hashes[0] == flipped_hashes[1]
    assert std > 8
    assert similarity(thumb, thumb)["rgb_mae"] == 0
    assert similarity(thumb, flipped[:, ::-1])["gray_correlation"] > 0.999999


def test_jpeg_reencoding_survives_candidate_screen_but_different_pixels_do_not():
    rgb = picture()
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    with Image.open(buffer) as image:
        decoded = np.asarray(image.convert("RGB"))
    a, ha, _ = describe(rgb)
    b, hb, _ = describe(decoded)
    assert (int(ha[0]) ^ int(hb[0])).bit_count() <= 8
    score = similarity(a, b)
    assert score["gray_correlation"] >= 0.995 and score["rgb_mae"] <= 8
    c, _, _ = describe(picture(71))
    assert similarity(a, c)["gray_correlation"] < 0.5


def test_blocked_hamming_screen_matches_exhaustive_integer_calculation():
    a = np.array([0, 1, 65535, 2**64-1], np.uint64)
    b = np.array([0, 3, 65534, 2**64-2, 1 << 40], np.uint64)
    actual = sorted(candidates(a, b, max_distance=2, block=2))
    expected = [(i, j, (int(x)^int(y)).bit_count()) for i, x in enumerate(a)
                for j, y in enumerate(b) if (int(x)^int(y)).bit_count() <= 2]
    assert actual == expected
    assert list(candidates(a[:0], b)) == []


def test_constant_image_is_flagged_and_bad_shapes_fail():
    _, _, std = describe(np.full((192, 192, 3), 128, np.uint8))
    assert std == 0
    with pytest.raises(ValueError):
        describe(np.zeros((192, 192), np.uint8))
    with pytest.raises(ValueError):
        list(candidates(np.array([-1]), np.array([1], np.uint64)))


def test_all_low_contrast_rows_produce_an_empty_screen():
    from skin_image_similarity import search

    rows = [dict(source="lapa", role="train"), dict(source="celeba", role="validation")]
    outcome = search(rows, np.zeros((2, 64, 64, 3), np.uint8),
                     np.zeros((2, 2), np.uint64), np.zeros(2))
    assert outcome["low_contrast_rows"] == 2
    assert outcome["searched_image_pairs"] == 0
    assert outcome["candidate_pairs"] == 0
