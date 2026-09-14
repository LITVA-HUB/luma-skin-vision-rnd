"""Matched CPU trajectories before any new head-range data experiment."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_architecture_scale import fit as original_fit
from chromaseed_head_range_fit import fit
from test_chromaseed_architecture_scale import fixture


@pytest.mark.parametrize("variant", ("patch_small", "soft_small", "dynamic_small"))
def test_unit_trajectory_and_short_prefix_are_bitwise_original(variant):
    x, tokens, y, warm = fixture()
    weights = np.linspace(0.25, 2, len(y))
    old, old_info = original_fit(x, tokens, y, weights, warm, variant, 8, (4, 8))
    new, info = fit(x, tokens, y, weights, warm, variant, "unit", 8, (4, 8))
    prefix, _ = fit(x, tokens, y, weights, warm, variant, "unit", 4, (4,))
    for step in (4, 8):
        assert len(new[step]) == len(old[step]) == 6
        for expected, actual in zip(old[step], new[step], strict=True):
            assert expected.keys() == actual.keys()
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
    for expected, actual in zip(new[4], prefix[4], strict=True):
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    for key in old_info.keys() - {"setup_seconds", "full_bank_seconds", "trace"}:
        assert info[key] == old_info[key]
    assert info["head_mode"] == "unit"
    for a, b in zip(info["trace"], old_info["trace"], strict=True):
        assert a["step"] == b["step"]
        np.testing.assert_array_equal(a["minibatch_loss"], b["minibatch_loss"])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mode", ("wide", "linear"))
def test_changed_head_fit_exports_its_mode_and_retains_original_sampling(mode):
    x, tokens, y, warm = fixture()
    weights = np.ones(len(y))
    unit, control = fit(x, tokens, y, weights, warm, "patch_small", "unit", 8, (8,))
    models, info = fit(x, tokens, y, weights, warm, "patch_small", mode, 8, (8,))
    assert info["head_mode"] == mode
    assert info["sampling_sha256"] == control["sampling_sha256"]
    assert info["final_learning_rates"] == control["final_learning_rates"]
    assert info["trainable_parameters"] == control["trainable_parameters"]
    for model, reference in zip(models[8], unit[8], strict=True):
        assert str(model["head_mode"]) == mode
        assert np.isfinite(model["theta"]).all()
        assert np.count_nonzero(model["theta"][-3:]) > 0
        for key in reference.keys() - {"theta"}:
            np.testing.assert_array_equal(model[key], reference[key])
    assert not torch.cuda.is_initialized()
