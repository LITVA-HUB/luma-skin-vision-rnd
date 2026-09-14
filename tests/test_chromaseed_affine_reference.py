"""QR compression preserves a separately constructed augmented coupled objective."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_affine_reference import qr_ridge
from chromaseed_perceptual_reference import augmented_svd


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_augmented_qr_svd_matches_full_coupled_svd(alpha):
    rng = np.random.default_rng(917372)
    design = rng.normal(size=(171, 11))
    design[:, -1] = design[:, 0] + 1e-9 * design[:, 2]
    y = rng.normal(size=(171, 3))
    weights = rng.uniform(0.001, 2, 171)
    metric = np.array([[1.4, 0.2, -0.3], [0.2, 0.5, 0.07], [-0.3, 0.07, 1.2]])
    expected = augmented_svd(design, y, np.broadcast_to(metric, (171, 3, 3)), weights, alpha)
    actual = qr_ridge(design, y, weights, metric, alpha)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
