"""Independent active-neuron contractions agree with the executable consumer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_patch8_fit import token_normalizers
from chromaseed_widen import Bank, predict
from chromaseed_widen_audit import direct, local_transform
from test_chromaseed_patch8 import fixture, independent_tokens


@pytest.mark.parametrize("variant", ["tiny", "m31", "m61", "m111", "m832"])
def test_independent_contraction_with_all_branches_active(variant):
    x, t, _, warm = fixture(5)
    m = Bank(warm, variant).export(0, warm, token_normalizers(t))
    rng = np.random.default_rng(235)
    m["g"][:] = rng.normal(0, 0.02, m["g"].shape).astype(np.float32)
    m["v0"][:] = rng.normal(0, 0.03, m["v0"].shape).astype(np.float32)
    np.testing.assert_allclose(direct(m, x, t), predict(m, x, t), rtol=0, atol=2e-8)


def test_independent_affine_transform_matches_pixel_statistics():
    rgb = np.random.default_rng(273).uniform(size=(128, 128, 3))
    tokens = independent_tokens(rgb).astype(np.float32)[None]
    anchor = np.array([0.2, 0.6, 0.8])
    dose = 64 / 255
    expected = independent_tokens(rgb + dose * (anchor - rgb)).astype(np.float32)
    np.testing.assert_allclose(
        local_transform(tokens, dose, anchor)[0], expected, rtol=0, atol=6e-8
    )
