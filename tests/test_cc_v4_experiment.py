"""Protocol tests; not experimental accuracy evidence."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/cc_v4_experiment.py"
    assert path.exists(), "V4 runner is not implemented"
    spec = importlib.util.spec_from_file_location("cc_v4_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_design_uses_only_detached_prediction_and_independent_rng():
    r = runner()
    point = torch.tensor([[.2, -.1], [-.3, .4]], requires_grad=True)
    a = r.sample_actions(point, torch.Generator().manual_seed(19))
    b = r.sample_actions(point, torch.Generator().manual_seed(19))
    assert torch.equal(a, b) and not a.requires_grad
    assert a.shape == (2, 33, 2)
    assert torch.equal(a[:, 0], point.detach())
    assert (a.abs() <= 2).all()
    assert ((a[:, 1:17] - point.detach()[:, None]).abs() <= .400001).all()


def test_repeated_refinement_diagnostics_count_true_regressions():
    r = runner()
    errors = np.array([[2., 1., 3., 4.], [3., 2., 2., 1.]])
    risks = np.array([[3., 2., 1., .5], [4., 3., 2., 1.]])
    result = r.refinement_summary(errors, risks)
    assert result["1_to_2"]["true_error_worsened"] == 0
    assert result["2_to_4"]["true_error_worsened"] == 1
    assert result["2_to_4"]["predicted_risk_increased"] == 0
    assert result["2_to_4"]["mean_true_delta"] == pytest.approx(1.)


def test_fixed_coverage_is_actual_integer_population_and_finite():
    r = runner()
    errors = np.arange(1., 120.)
    result = r.risk_summary(errors, -errors)
    row = result["fixed"]["80"]
    assert row["accepted"] == 95 and row["coverage"] == pytest.approx(95 / 119)
    assert row["mean"] > errors.mean()
    assert len(result["curve"]) == 119
    with pytest.raises(ValueError):
        r.risk_summary(errors, np.full(119, np.nan))


def test_zero_field_warmup_and_later_fixed_schedule():
    r = runner()
    assert r.field_weight(0) == 0 and r.field_weight(20) == 0
    assert r.field_weight(30) == .5 and r.field_weight(40) == 1
    assert r.field_weight(120) == 1
