"""Numerical oracles for unpenalized-bias compact neural readouts."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_gaussian import export, initialize, predict, prepare
from chromaseed_gaussian_numpy import Predictor
from chromaseed_neural_readout import fit_head, fit_single, representation_bank, solve


def fixture(group="raw36", n=31, hidden=8):
    rng = np.random.default_rng(931)
    x = rng.uniform(0.1, 0.9, (n, 36)).astype(np.float32)
    y = np.column_stack((25 + x[:, 27] * 60, 3 + x[:, 28] * 15, 8 + x[:, 29] * 22)).astype(float)
    prep, _, _ = prepare(x, y, group)
    source = export(prep, initialize(36 if group == "raw36" else 3, 17, hidden)[0])
    return source, x, y, rng.uniform(0.3, 2, n)


def augmented(z, t, metric, w, alpha):
    rank = z.shape[1]
    root = np.linalg.cholesky(metric).transpose(0, 2, 1) * np.sqrt(w)[:, None, None]
    a = np.einsum("nab,nc->nacb", root, z).reshape(3 * len(z), 3 * rank)
    b = np.einsum("nab,nb->na", root, t).ravel()
    penalty = np.eye(3 * rank) * np.sqrt(alpha)
    penalty[:3, :3] = 0
    return np.linalg.lstsq(np.vstack((a, penalty)), np.r_[b, np.zeros(3 * rank)], rcond=None)[
        0
    ].reshape(rank, 3)


def test_unpenalized_intercept_matches_scalar_augmented_lstsq():
    z = np.column_stack((np.ones(7), np.linspace(-2, 3, 7)))
    target = np.column_stack((3 + z[:, 1], -5 + 2 * z[:, 1], 8 - z[:, 1]))
    w = np.arange(1, 8, dtype=float)
    alpha = 4.0
    expected = augmented(z, target, np.broadcast_to(np.eye(3), (7, 3, 3)), w, alpha)
    beta, residual = solve(z, target, None, w, alpha)
    np.testing.assert_allclose(beta, expected, atol=1e-12)
    assert residual < 1e-12
    shift = np.array([100, -70, 45])
    shifted, _ = solve(z, target + shift, None, w, alpha)
    np.testing.assert_allclose(shifted[0] - beta[0], shift, atol=1e-12)
    np.testing.assert_allclose(shifted[1:], beta[1:], atol=1e-12)


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_correlated_output_metric_matches_independent_augmented_lstsq(alpha):
    rng = np.random.default_rng(190)
    z = np.column_stack((np.ones(19), rng.normal(size=(19, 5))))
    t, w = rng.normal(size=(19, 3)), rng.uniform(0.2, 2, 19)
    c = rng.normal(size=(19, 3, 3))
    metric = c @ c.transpose(0, 2, 1) + np.eye(3)[None] * 0.2
    beta, residual = solve(z, t, metric, w, alpha)
    np.testing.assert_allclose(beta, augmented(z, t, metric, w, alpha), atol=2e-12)
    assert residual < 1e-12


@pytest.mark.parametrize("group", ["raw36", "mean3"])
@pytest.mark.parametrize("family", ["norm", "perceptual"])
def test_folded_network_preserves_representation_and_actual_consumer(group, family):
    source, x, y, w = fixture(group)
    original = {k: v.copy() for k, v in source.items()}
    model, info = fit_head(source, x, y, w, family, 0.1)
    for k in source:
        np.testing.assert_array_equal(source[k], original[k])
        if k not in ("w2", "b2"):
            np.testing.assert_array_equal(model[k], source[k])
    assert set(model) == set(source)
    assert sum(v.nbytes for v in model.values()) == sum(v.nbytes for v in source.values())
    assert all(v.dtype == original[k].dtype for k, v in model.items())
    assert info["max_folded_lab_drift"] < 0.001
    assert info["normal_residual"] < 1e-10
    consumer = Predictor(model)
    actual = np.array([consumer(row) for row in x])
    np.testing.assert_allclose(actual, predict(model, x), atol=1e-10)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("family", ["norm", "perceptual"])
def test_all_dead_hidden_channels_fit_an_unpenalized_constant(family):
    source, x, y, w = fixture()
    source["w1"][:] = 0
    source["b1"][:] = -1
    model, info = fit_head(source, x, y, w, family, 10)
    assert info["hidden_constant_columns"] == 8
    np.testing.assert_allclose(model["w2"], 0, atol=1e-12)
    assert np.isfinite(predict(model, x)).all()
    if family == "norm":
        np.testing.assert_allclose(
            predict(model, x), np.broadcast_to(np.average(y, weights=w, axis=0), y.shape), atol=2e-6
        )


def test_mean3_ignores_finite_other_features():
    source, x, y, w = fixture("mean3")
    model, _ = fit_head(source, x, y, w, "norm", 1)
    changed = x.copy()
    changed[:, :27] = 12
    changed[:, 30:] = -8
    consumer = Predictor(model)
    np.testing.assert_array_equal([consumer(row) for row in x], [consumer(row) for row in changed])


@pytest.mark.parametrize("bad", ["alpha", "weight", "family", "shape", "nan"])
def test_invalid_fit_inputs_are_rejected(bad):
    source, x, y, w = fixture()
    alpha, family = 1, "norm"
    if bad == "alpha":
        alpha = 0
    elif bad == "weight":
        w[0] = -1
    elif bad == "family":
        family = "unknown"
    elif bad == "shape":
        y = y[:, :2]
    else:
        x[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_head(source, x, y, w, family, alpha)


def test_fixed_representation_banks_match_single_full_construction():
    _, x, y, _ = fixture("mean3", n=9)
    person, site = np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)
    from skin_local_search_train import weights_for

    bank, trajectories = representation_bank(
        x, y, weights_for(person, site), "mean3", seeds=(17,), epochs=(1, 4)
    )
    assert len(bank) == 5 and len(trajectories) == 2
    for basis in ("random", "adam_e1", "adam_e4", "tagi_full3_e1", "tagi_full3_e4"):
        expected, _ = fit_head(bank[(basis, 17)], x, y, weights_for(person, site), "norm", 1)
        actual, _ = fit_single(x, y, person, site, "mean3", basis, "norm", 1, 17)
        for k in expected:
            np.testing.assert_array_equal(expected[k], actual[k])
        assert sum(v.nbytes for v in actual.values()) == 1855


@pytest.mark.parametrize("group", ["raw36", "mean3"])
@pytest.mark.parametrize("family", ["norm", "perceptual"])
def test_independent_qr_head_matches_native_predictions(group, family):
    from chromaseed_neural_readout_reference import refit

    source, x, y, w = fixture(group, n=43, hidden=64)
    fitted, _ = fit_head(source, x, y, w, family, 0.1)
    independent, info = refit(source, x, y, w, family, 0.1)
    np.testing.assert_allclose(predict(fitted, x), predict(independent, x), atol=0.001, rtol=0)
    assert info["rank"] == 65 * (3 if family == "perceptual" else 1)
    for k in source:
        if k not in ("w2", "b2"):
            np.testing.assert_array_equal(fitted[k], independent[k])


def test_selection_ties_prefer_declared_alpha_and_shorter_representation():
    from chromaseed_neural_readout import choose, choose_policy

    a = [dict(clean=2, p90=4, alpha_index=i) for i in (2, 1, 0)]
    assert choose(a)["alpha_index"] == 0
    b = [dict(clean=2, p90=4, basis=v) for v in ("tagi_full3_e16", "adam_e1", "random")]
    assert choose_policy(b)["basis"] == "random"


@pytest.mark.parametrize("group", ["raw36", "mean3"])
def test_registered_representation_prefixes_match_independent_scalar_training(group):
    from chromaseed_gaussian_reference import train_reference

    _, x, y, weights = fixture(group, n=9)
    bank, _ = representation_bank(x, y, weights, group, seeds=(17,))
    for method, parameter in (("adam", 0.001), ("tagi_full3", 1.0)):
        independent, _ = train_reference(
            x, y, weights, group, method, parameter, 17, checkpoints=(1, 4, 16)
        )
        for epoch in (1, 4, 16):
            for k, value in independent[epoch].items():
                np.testing.assert_allclose(
                    bank[(f"{method}_e{epoch}", 17)][k], value, atol=2e-6, rtol=2e-6
                )
