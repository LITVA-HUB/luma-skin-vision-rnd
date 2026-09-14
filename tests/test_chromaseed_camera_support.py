import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_camera_person_site_weights_and_person_reduction():
    from chromaseed_camera_support import aggregate_people, row_weights

    p = np.array([0, 0, 0, 1, 2, 2])
    s = np.array([0, 0, 1, 0, 0, 0])
    c = np.array([0, 0, 0, 0, 1, 1])
    w = row_weights(p, s, c)
    np.testing.assert_allclose(w, [0.0625, 0.0625, 0.125, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(
        aggregate_people(np.array([0.0, 2.0, 5.0, 7.0, 0.0, 4.0]), p, s), [3.0, 7.0, 2.0]
    )


def test_fit_standardizer_does_not_see_query_extremes():
    from chromaseed_camera_support import standardize

    x = np.array([[0.0, 1.0], [2.0, 1.0]])
    q = np.array([[1e9, 4.0]])
    z, v, prep = standardize(x, q, np.array([0.5, 0.5]))
    np.testing.assert_array_equal(z, [[-1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(prep["mean"], [1.0, 1.0])
    np.testing.assert_allclose(prep["std"], [1.0, 1e-6])
    assert v[0, 0] == 1e9 - 1


def test_linear_and_kernel_scores_match_independent_lstsq():
    from chromaseed_camera_support import classify, kernel

    rng = np.random.default_rng(217)
    x = rng.normal(size=(21, 5))
    q = rng.normal(size=(7, 5))
    y = rng.choice([-1.0, 1.0], 21)
    w = rng.uniform(0.2, 1, 21)
    w /= w.sum()
    actual, info = classify(x, q, y, w, "linear")
    design = np.column_stack([np.ones(21), x])
    penalty = np.column_stack([np.zeros(5), np.sqrt(0.1) * np.eye(5)])
    coefficient = np.linalg.lstsq(
        np.vstack([np.sqrt(w)[:, None] * design, penalty]),
        np.r_[np.sqrt(w) * y, np.zeros(5)],
        rcond=None,
    )[0]
    np.testing.assert_allclose(
        actual, np.column_stack([np.ones(7), q]) @ coefficient, atol=1e-11, rtol=0
    )
    actual, info = classify(x, q, y, w, "rbf")
    k = kernel(x, x, info["width"])
    root = np.sqrt(w)
    coeff = (
        np.linalg.lstsq(
            root[:, None] * k * root[None, :] + 0.01 * np.eye(21), root * y, rcond=None
        )[0]
        * root
    )
    np.testing.assert_allclose(actual, kernel(q, x, info["width"]) @ coeff, atol=1e-11, rtol=0)


def test_linear_target_residual_is_orthogonal_on_fit_only():
    from chromaseed_camera_support import views

    rng = np.random.default_rng(307)
    y = rng.normal(size=(25, 3))
    x = y @ rng.normal(size=(3, 36)) + rng.normal(size=(25, 36)) * 0.02
    w = np.ones(25) / 25
    fit, query = views(x, y, x[:2], y[:2] + 100.0, w)
    assert list(fit) == ["color36", "rgb_mean", "lab3", "lab_residual"]
    np.testing.assert_allclose(
        fit["lab3"].T @ (w[:, None] * fit["lab_residual"]), 0.0, atol=1e-10, rtol=0
    )
    assert np.abs(query["lab_residual"]).max() > 100.0


def test_auc_ties_and_person_counts():
    from chromaseed_camera_support import classification_metrics

    m = classification_metrics(np.array([0.0, 1.0, 0.0, 0.0]), np.array([1, 1, -1, -1]))
    assert m["auc"] == 0.75 and m["balanced_accuracy"] == 0.5
    assert m["slr_correct"] == 2 and m["ipod_correct"] == 0
    assert classification_metrics(np.array([]), np.array([])) is None


def test_assignment_maximizes_cardinality_before_distance():
    from chromaseed_camera_support import match_cost

    row, col = match_cost(np.array([[1.0, 2.0], [1.1, 100.0]]), 2.0)
    np.testing.assert_array_equal(row, [0, 1])
    np.testing.assert_array_equal(col, [1, 0])
    row, col = match_cost(np.array([[1.0, 2.0], [1.1, 100.0]]), 0.5)
    assert len(row) == len(col) == 0


def test_support_excludes_every_row_of_same_person():
    from chromaseed_camera_support import nearest

    x = np.array([[0.0], [0.0], [2.0], [2.0]])
    person = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(nearest(x, x, "rms", person, person), [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(nearest(np.array([[1.0]]), x, "rms"), [1.0])


def test_weighted_empirical_quantile_uses_cumulative_mass():
    from chromaseed_camera_support import weighted_quantile

    x = np.array([100.0, 0.0, 10.0])
    w = np.array([0.02, 0.90, 0.08])
    assert weighted_quantile(x, w, 0.95) == 10.0
    assert weighted_quantile(x, w, 0.5) == 0.0


@pytest.mark.parametrize(
    "bad", [np.array([np.nan, 1.0]), np.array([-1.0, 2.0]), np.array([0.0, 0.0])]
)
def test_invalid_normalization_weights_fail(bad):
    from chromaseed_camera_support import standardize

    with pytest.raises(ValueError):
        standardize(np.ones((2, 3)), np.ones((1, 3)), bad)
