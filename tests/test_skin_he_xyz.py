import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from cc_v7_external_audit import angle, percentile
from skin_he_xyz import inputs, metrics, model


def test_linear_calibration_recovers_known_cross_channel_mapping():
    x = np.random.default_rng(17).uniform(.1, 1, (30, 3))
    mapping = np.array([[3, 1, .2], [.1, 2, .3], [.5, .2, 4]])
    fitted = model("linear3").fit(x[:20], x[:20] @ mapping)
    np.testing.assert_allclose(fitted.predict(x[20:]), x[20:] @ mapping, atol=1e-12)


def test_root_features_preserve_scalar_exposure():
    x = np.array([[.2, .3, .4], [.1, .8, .3]])
    np.testing.assert_allclose(inputs(x*7, "root2"), inputs(x, "root2")*7)


def test_xyz_error_is_coordinate_error_with_brightness_sensitivity():
    gt = np.array([[10., 20., 30.], [10., 20., 30.]])
    pred = gt + [3, 4, 0]
    report = metrics(pred, gt)
    assert report["xyz_rmse"] == pytest.approx(5/np.sqrt(3))
    assert report["median_euclidean_xyz"] == 5
    assert metrics(gt*2, gt)["xyz_rmse"] > 0


def test_independent_atan2_handles_parallel_and_perpendicular_vectors():
    assert angle([1, 2, 3], [2, 4, 6]) == 0
    assert angle([1, 0, 0], [0, 1, 0]) == 90
    assert percentile([0., 2., 6., 8.], .25) == 1.5
