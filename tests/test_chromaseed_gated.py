import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def sample():
    rng = np.random.default_rng(913)
    x = rng.normal(size=(65, 36)).astype(np.float32)
    y = rng.normal([51.0, 10.0, 18.0], [9.0, 4.0, 5.0], (65, 3))
    p, s = np.repeat(np.arange(13), 5), np.tile([0, 0, 1, 1, 2], 13)
    c = np.where(p < 5, "SLR", "ipod")
    return x, y, p, s, c


@pytest.mark.parametrize("base,old", [("norm", "norm_mse"), ("perceptual", "constant_de2")])
def test_zero_and_base_payloads_preserve_frozen_p(base, old):
    from chromaseed_gated import fit_single
    from chromaseed_perceptual import fit_single as pfit
    from skin_local_search_train import weights_for

    x, y, p, s, c = sample()
    w = weights_for(p, s)
    expected, _ = pfit(x, y, w, old, 17, 1, 0, 0 if base == "norm" else 1, rank=16)
    for route in ("base", "uniform", "soft", "hard"):
        actual, _ = fit_single(x, y, p, s, c, f"{base}_{route}", 17, 0.1, 0.0, rank=16)
        assert set(actual) == set(expected)
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("route", ["soft", "hard"])
def test_one_camera_routed_fallback_is_exact_but_uniform_can_fit(route):
    from chromaseed_gated import fit_single

    x, y, p, s, c = sample()
    c[:] = "SLR"
    expected, _ = fit_single(x, y, p, s, c, "norm_base", 17, 0.1, 0.0, rank=16)
    actual, info = fit_single(x, y, p, s, c, f"norm_{route}", 17, 0.1, 1.0, rank=16)
    assert info["fallback"] == "single_camera" and info["residual_solutions"] == 0
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    ordinary, info = fit_single(x, y, p, s, c, "norm_uniform", 17, 0.1, 1.0, rank=16)
    assert info["residual_solutions"] == 1 and not np.array_equal(
        ordinary["coefficient"], expected["coefficient"]
    )


def test_coupled_residual_bank_matches_augmented_svd():
    from chromaseed_gated import residual_solutions
    from chromaseed_perceptual_reference import augmented_svd

    rng = np.random.default_rng(312)
    z = rng.normal(size=(49, 9))
    z[:, 8] = z[:, 0] + 1e-7 * z[:, 8]
    y = rng.normal(size=(49, 3))
    w = rng.uniform(0.2, 2.0, 49)
    a = rng.normal(size=(3, 3))
    g = a @ a.T + 0.2 * np.eye(3)
    theta, info = residual_solutions(z, y, g, w, (0.1, 1.0, 10.0))
    for alpha in theta:
        reference = augmented_svd(z, y, np.broadcast_to(g, (49, 3, 3)), w, alpha)
        np.testing.assert_allclose(z @ theta[alpha], z @ reference, atol=1e-10, rtol=0)
    assert max(d["objective_minus_zero"] for d in info["solutions"]) <= 1e-10


def test_gate_clipping_and_exact_hard_boundary():
    from chromaseed_gated import gate_value

    z = np.zeros((5, 36))
    z[:, 0] = [-2.0, -0.1, 0.0, 0.1, 2.0]
    beta = np.r_[0.0, 1.0, np.zeros(35)].astype(np.float32)
    np.testing.assert_array_equal(gate_value(z, beta, "soft"), [-1.0, -0.1, 0.0, 0.1, 1.0])
    np.testing.assert_array_equal(gate_value(z, beta, "hard"), [-1.0, -1.0, 1.0, 1.0, 1.0])


def test_static_collapse_and_dynamic_predictor_share_one_basis():
    from chromaseed_gated import package_model, predict
    from chromaseed_kernel import coordinates, gaussian_kernel, predict_kernel

    x, _, _, _, _ = sample()
    base = dict(
        x_mean=np.zeros(36, np.float32),
        x_std=np.ones(36, np.float32),
        y_mean=np.zeros(3, np.float32),
        y_std=np.ones(3, np.float32),
        centers=x[:7],
        width=np.array(1.0, np.float32),
        coefficient=np.ones((7, 3), np.float32),
    )
    correction = np.ones((7, 3), np.float32) * 0.25
    beta = np.r_[0.0, 1.0, np.zeros(35)].astype(np.float32)
    static = package_model(base, correction, None, "uniform", 0.5)
    assert set(static) == set(base)
    np.testing.assert_array_equal(static["coefficient"], np.full((7, 3), 1.125, np.float32))
    dynamic = package_model(base, correction, beta, "hard", 0.5)
    z = coordinates(base, x)
    k = gaussian_kernel(z, base["centers"], 1.0)
    expected = predict_kernel(base, x) + 0.5 * np.where(z[:, 0] >= 0, 1.0, -1.0)[:, None] * (
        k @ correction
    )
    np.testing.assert_allclose(predict(dynamic, x), expected, atol=1e-12, rtol=0)
    assert (
        sum(v.nbytes for v in dynamic.values()) - sum(v.nbytes for v in base.values())
        == 7 * 3 * 4 + 37 * 4 + 4 + 1
    )


def test_registered_candidate_grid_and_invalid_settings():
    from chromaseed_gated import FAMILIES, candidates, fit_single

    assert len(FAMILIES) == 8
    assert sum(len(candidates(f)) for f in FAMILIES) == 62
    x, y, p, s, c = sample()
    with pytest.raises(ValueError):
        fit_single(x, y, p, s, c, "norm_soft", 17, -1.0, 1.0, rank=16)
