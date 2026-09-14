"""Numerical invariants for legal pixel contractions and gate boundaries."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_gate_stability import affine_features, feature_direction, gate_score, projection
from chromaseed_gated_numpy import Predictor


def stats(pixels):
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    covariance = (pixels - mean).T @ (pixels - mean) / len(pixels)
    corr = covariance / np.maximum(std[:, None] * std[None, :], 1e-12)
    corr[np.ptp(pixels, axis=0) == 0, :] = 0
    corr[:, np.ptp(pixels, axis=0) == 0] = 0
    return np.r_[
        np.quantile(pixels, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], axis=0).ravel(),
        mean,
        std,
        corr[0, 1],
        corr[0, 2],
        corr[1, 2],
    ]


def model():
    return dict(
        x_mean=np.zeros(36, np.float32),
        x_std=np.ones(36, np.float32),
        y_mean=np.array([50, 10, 10], np.float32),
        y_std=np.ones(3, np.float32),
        centers=np.zeros((1, 36), np.float32),
        coefficient=np.zeros((1, 3), np.float32),
        width=np.array(1, np.float32),
        correction=np.array([[3, 0, 0]], np.float32),
        gate_beta=np.r_[-0.5, 1, np.zeros(35)].astype(np.float32),
        gate_mode=np.array(2, np.uint8),
        rho=np.array(0.5, np.float32),
    )


@pytest.mark.parametrize("t", [1 / 255, 16 / 255, 64 / 255])
def test_affine_matches_real_pixel_statistics(t):
    rng = np.random.default_rng(917)
    pixels = rng.uniform(0, 1, (151, 3))
    pixels[:, 2] = 0.25
    anchor = np.array([1, 0, 1])
    expected = stats((1 - t) * pixels + t * anchor)
    actual = affine_features(stats(pixels)[None], t, anchor)[0]
    np.testing.assert_allclose(actual, expected, atol=8e-8, rtol=0)
    assert actual[32] == 0 and actual[34] == actual[35] == 0


def test_identity_and_bounded_quantiles():
    x = stats(np.random.default_rng(5).uniform(0, 1, (101, 3))).astype(np.float32)[None]
    np.testing.assert_array_equal(affine_features(x, 0, [0, 0, 0]), x)
    for anchor in ([0, 0, 0], [1, 1, 1], [1, 0, 1]):
        out = affine_features(x, 0.99, anchor)
        assert (out[:, :30] >= 0).all() and (out[:, :30] <= 1).all()
        assert (np.diff(out[:, :27].reshape(-1, 9, 3), axis=1) >= 0).all()


def test_bad_transform_rejected():
    x = np.zeros((1, 36))
    for t in (-0.01, 1, np.nan):
        with pytest.raises(ValueError):
            affine_features(x, t, [0, 1, 0])
    with pytest.raises(ValueError):
        affine_features(x, 0.1, [0, 2, 0])


def test_affine_root_predicts_actual_gate_crossing():
    m = model()
    x = np.full((1, 36), 0.2, dtype=np.float32)
    x[:, 30:33] = 0.1
    slope = (feature_direction(x, [1, 0, 0]) / m["x_std"]) @ m["gate_beta"][1:]
    root = -gate_score(m, x) / slope
    assert root[0] == pytest.approx(0.375, abs=1e-7)
    assert gate_score(m, affine_features(x, float(root[0] - 1e-4), [1, 0, 0]))[0] < 0
    assert gate_score(m, affine_features(x, float(root[0] + 1e-4), [1, 0, 0]))[0] > 0


def test_projection_is_nearest_hyperplane_point():
    m = model()
    m["gate_beta"][2] = 2
    x = np.full((2, 36), 0.2, np.float32)
    z, score, projected, rms = projection(m, x)
    beta = m["gate_beta"].astype(np.float64)
    np.testing.assert_allclose(projected @ beta[1:] + beta[0], 0, atol=1e-15)
    np.testing.assert_allclose(
        np.linalg.norm(projected - z, axis=1), np.abs(score) / np.linalg.norm(beta[1:]), atol=1e-15
    )
    np.testing.assert_allclose(rms, np.linalg.norm(projected - z, axis=1) / 6, atol=1e-15)
    tangent = np.r_[2, -1, np.zeros(34)]
    assert np.all(np.linalg.norm(projected + tangent - z, axis=1) > rms * 6)


def test_hard_limit_matches_actual_two_sided_prediction():
    m = model()
    p = Predictor(m)
    x = np.zeros(36, np.float32)
    x[0] = 0.5
    left, right = x.copy(), x.copy()
    left[0] -= 1e-5
    right[0] += 1e-5
    observed = p(right) - p(left)
    expected = 2 * float(m["rho"]) * np.exp(-0.5 * 0.25 / 36) * m["correction"][0]
    np.testing.assert_allclose(observed, expected, atol=2e-7, rtol=0)
