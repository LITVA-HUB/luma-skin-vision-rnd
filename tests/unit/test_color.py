from pathlib import Path

import numpy as np
import pytest

from luma_skin_vision.color import delta_e00, lab_to_srgb, srgb_to_lab


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([50, 2.6772, -79.7751], [50, 0, -82.7485], 2.0425),
        ([50, 3.1571, -77.2803], [50, 0, -82.7485], 2.8615),
        ([50, 2.8361, -74.0200], [50, 0, -82.7485], 3.4412),
        ([50, -1.3802, -84.2814], [50, 0, -82.7485], 1.0000),
        ([50, -1.1848, -84.8006], [50, 0, -82.7485], 1.0000),
        ([50, -0.9009, -85.5211], [50, 0, -82.7485], 1.0000),
        ([50, 0, 0], [50, -1, 2], 2.3669),
        ([50, 2.49, -0.001], [50, -2.49, 0.0009], 7.1792),
        ([50, 2.49, -0.001], [50, -2.49, 0.0011], 7.2195),
    ],
)
def test_sharma_reference(a, b, expected):
    assert delta_e00(a, b) == pytest.approx(expected, abs=5e-5)
    assert delta_e00(b, a) == pytest.approx(expected, abs=5e-5)


def test_reference_colors_and_roundtrip():
    np.testing.assert_allclose(srgb_to_lab([1, 1, 1]), [100, 0, 0], atol=2e-5)
    np.testing.assert_allclose(srgb_to_lab([0, 0, 0]), [0, 0, 0], atol=1e-8)
    np.testing.assert_allclose(srgb_to_lab([1, 0, 0]), [53.2408, 80.0925, 67.2032], atol=1e-3)
    rgb = np.random.default_rng(0).uniform(size=(50, 3))
    np.testing.assert_allclose(lab_to_srgb(srgb_to_lab(rgb)), rgb, atol=2e-6)
    np.testing.assert_allclose(delta_e00(srgb_to_lab(rgb), srgb_to_lab(rgb)), 0)


def test_reject_invalid_color():
    with pytest.raises(ValueError):
        srgb_to_lab([255, 0, 0])
    with pytest.raises(ValueError):
        delta_e00([np.nan, 1, 2], [0, 1, 2])


def test_all_34_sharma_supplementary_pairs():
    rows = np.loadtxt(Path(__file__).parents[1] / "fixtures" / "ciede2000_sharma.txt")
    assert len(rows) == 34
    np.testing.assert_allclose(delta_e00(rows[:, :3], rows[:, 3:6]), rows[:, 6], atol=5e-5, rtol=0)
