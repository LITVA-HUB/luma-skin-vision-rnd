import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chromaseed import color36
from chromaseed_affine import fit_single as a_fit
from chromaseed_feature_groups import (
    GROUPS,
    choose_alpha,
    choose_policy,
    fit_bank,
    fit_single,
    gather,
    model_id,
    predict,
)
from chromaseed_feature_groups_numpy import Predictor
from chromaseed_projection_reference import predict as independent_predict
from chromaseed_projection_reference import refit


@pytest.fixture
def toy():
    rng = np.random.default_rng(70513)
    rgb = rng.uniform(0.1, 0.9, (60, 1, 3))
    pixels = np.clip(rgb + rng.normal(0, 0.05, (60, 32, 3)), 0, 1)
    x = color36(pixels).astype(np.float32)
    y = np.column_stack([20 + 60 * x[:, 27], 5 + 15 * x[:, 28], 8 + 20 * x[:, 29]])
    person = np.repeat(np.arange(10), 6)
    site = np.arange(60) // 2
    camera = np.where(person < 5, "SLR", "ipod")
    return x, y.astype(np.float64), person, site, camera


def exact(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k].dtype == b[k].dtype
        np.testing.assert_array_equal(a[k], b[k])


def test_groups_preserve_quantile_major_channel_order():
    x = np.arange(72, dtype=np.float32).reshape(2, 36)
    expected = {
        "raw36": list(range(36)),
        "mean3": [27, 28, 29],
        "median3": [12, 13, 14],
        "central9": list(range(9, 18)),
        "mean_std6": list(range(27, 33)),
        "quant27": list(range(27)),
        "no_corr33": list(range(33)),
    }
    assert GROUPS == expected
    for name, indices in expected.items():
        np.testing.assert_array_equal(gather(x, name), x[:, indices])


@pytest.mark.parametrize("family", ["norm_static", "perceptual_joint_soft"])
def test_raw36_is_exact_A_fit(toy, family):
    expected, _ = a_fit(*toy, family, 17, 0.1, 0, rank=16)
    actual, _ = fit_single(*toy, family, 17, "raw36", 0.1, rank=16)
    exact(actual, expected)


@pytest.mark.parametrize(
    "group", ["mean3", "median3", "central9", "mean_std6", "quant27", "no_corr33"]
)
def test_reduced_fit_matches_independent_QR_and_numpy(toy, group):
    x, y, p, s, c = toy
    model, _ = fit_single(*toy, "perceptual_joint_soft", 17, group, 0.1, rank=16)
    ids = model["feature_indices"]
    reference, info = refit(
        model, x[:, ids], y, p, s, c, "perceptual_joint_soft", 0.1, "raw", 17, rank=16
    )
    expected = independent_predict(reference, x[:, ids])
    np.testing.assert_allclose(predict(model, x), expected, atol=1e-3, rtol=0)
    np.testing.assert_allclose(
        [Predictor(model)(row) for row in x], predict(model, x), atol=2e-8, rtol=0
    )
    assert info["width_drift"] == 0
    changed = x.copy()
    excluded = sorted(set(range(36)) - set(ids.tolist()))
    changed[:, excluded] += 7
    np.testing.assert_array_equal(predict(model, changed), predict(model, x))
    np.testing.assert_array_equal(Predictor(model)(changed[0]), Predictor(model)(x[0]))


def test_single_camera_joint_alias_and_fit_bank_parity(toy):
    x, y, p, s, c = toy
    c = np.full(len(c), "SLR")
    bank, receipt = fit_bank(x, y, p, s, c, rank=12, groups=("mean3",), seeds=(17,))
    for loss in ("norm", "perceptual"):
        for alpha in (0.1, 1.0, 10.0):
            a = model_id(loss + "_static", 17, "mean3", alpha)
            b = model_id(loss + "_joint_soft", 17, "mean3", alpha)
            exact(bank[a], bank[b])
            one, _ = fit_single(x, y, p, s, c, loss + "_static", 17, "mean3", alpha, rank=12)
            exact(one, bank[a])
    assert receipt["new_coefficient_solutions"] == 6
    assert receipt["gate_fits"] == 0


def test_fit_target_changes_do_not_change_input_geometry(toy):
    x, y, p, s, c = toy
    a, _ = fit_single(*toy, "norm_joint_soft", 17, "mean3", 0.1, rank=12)
    b, _ = fit_single(x, y[::-1], p, s, c, "norm_joint_soft", 17, "mean3", 0.1, rank=12)
    for key in ("feature_indices", "x_mean", "x_std", "centers", "width", "gate_beta"):
        np.testing.assert_array_equal(a[key], b[key])


def test_consumer_rejects_wrong_feature_schema(toy):
    m, _ = fit_single(*toy, "norm_static", 17, "mean3", 0.1, rank=12)
    for wrong in (
        np.array([27, 27, 29], np.uint8),
        np.array([27, 28, 36], np.uint8),
        np.array([27, 28, 29], np.int64),
    ):
        with pytest.raises(ValueError):
            Predictor({**m, "feature_indices": wrong})
    with pytest.raises(ValueError):
        Predictor(m)(np.zeros(3))
    with pytest.raises(ValueError):
        Predictor(m)(np.full(36, np.nan))


def test_selection_uses_group_best_alpha_then_raw_allowances():
    def row(group, clean, p90, size, alpha=0.1, order=0):
        return dict(group=group, clean=clean, p90=p90, numeric_bytes=size, alpha=alpha, order=order)

    a = row("mean3", 5, 10, 20, alpha=0.1)
    b = row("mean3", 5, 10, 20, alpha=10.0)
    assert choose_alpha([a, b]) == b
    raw = row("raw36", 5, 10, 100)
    too_bad = row("median3", 5.051, 9, 5, order=1)
    tail_bad = row("central9", 4.9, 10.101, 6, order=2)
    allowed = row("mean3", 5.049, 10.099, 20, order=3)
    rows = [raw, too_bad, tail_bad, allowed]
    assert choose_policy(rows, "compact") == allowed
    assert choose_policy(rows, "quality") == tail_bad


def test_constant_consumer_is_portable():
    m = {"constant_lab": np.array([40.0, 15.0, 20.0], np.float32)}
    np.testing.assert_array_equal(Predictor(m)(np.zeros(36)), m["constant_lab"])
