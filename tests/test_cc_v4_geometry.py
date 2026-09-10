"""Constructed mathematical target checks; no new real benchmark evidence."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cc_v4_geometry import correction_targets


def test_exact_correction_has_zero_cost_and_gradient():
    gt = np.array([[0.7, 0.8, 0.3], [0.2, 0.5, 0.9]])
    action = np.log(gt[:, [0, 2]] / gt[:, 1:2])[:, None]
    result = correction_targets(gt, action)
    assert np.max(np.abs(result["sin2_cost"])) < 1e-14
    assert np.max(np.abs(result["cost_gradient"])) < 1e-14
    assert np.max(np.abs(result["reproduction_degrees"])) < 1e-12


def test_analytic_action_derivative_matches_finite_difference():
    rng = np.random.default_rng(11)
    gt = rng.uniform(0.02, 1, (4, 3))
    actions = rng.uniform(-1.5, 1.5, (4, 17, 2))
    result = correction_targets(gt, actions)
    for j in range(2):
        step = np.zeros_like(actions)
        step[..., j] = 1e-5
        numerical = (
            correction_targets(gt, actions + step)["sin2_cost"]
            - correction_targets(gt, actions - step)["sin2_cost"]
        ) / 2e-5
        assert np.allclose(numerical, result["cost_gradient"][..., j], atol=2e-9, rtol=0)
    assert np.allclose(
        np.sin(np.deg2rad(result["reproduction_degrees"])) ** 2, result["sin2_cost"], atol=1e-14
    )


def test_joint_gain_action_translation_and_gt_scale_invariance():
    gt = np.array([[0.4, 0.6, 0.9], [0.8, 0.7, 0.6]])
    a = np.array([[[0.1, -0.5]], [[0.3, 0.2]]])
    shift = np.array([[0.7, -0.4], [-0.8, 0.2]])
    gains = np.exp(np.column_stack((shift[:, 0], np.zeros(2), shift[:, 1])))
    base = correction_targets(gt, a)
    transformed = correction_targets(gt * gains, a + shift[:, None])
    scaled = correction_targets(gt * np.array([[1e-250], [1e250]]), a)
    for key in base:
        assert np.allclose(base[key], transformed[key], atol=1e-12, rtol=0)
        assert np.allclose(base[key], scaled[key], atol=1e-11, rtol=0)


def test_invalid_targets_refuse():
    with pytest.raises(ValueError):
        correction_targets(np.array([[0.0, 1.0, 1.0]]), np.zeros((1, 1, 2)))
    with pytest.raises(ValueError):
        correction_targets(np.ones((1, 3)), np.full((1, 1, 2), np.nan))
