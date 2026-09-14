"""CPU-only behavior of output-range changes, separate from running AS training."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_architecture_scale import VARIANTS
from chromaseed_architecture_scale import Bank as OriginalBank
from chromaseed_head_range import Bank, Predictor, head, predict_torch
from chromaseed_neural_prefix_numpy import predict as warm_predict
from test_chromaseed_architecture_scale import fixture


def test_head_range_and_identical_value_and_slope_at_zero():
    for mode in ("unit", "wide", "linear"):
        z = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        out = head(z, mode)
        np.testing.assert_array_equal(out.detach().numpy(), np.zeros(3))
        derivative = torch.autograd.grad(out.sum(), z)[0]
        np.testing.assert_array_equal(derivative.numpy(), np.ones(3))
    z = torch.linspace(-100, 100, 1001, dtype=torch.float64)
    assert head(z, "unit").abs().max() <= 1
    assert head(z, "wide").abs().max() <= 4
    assert head(z, "wide").abs().max() > 3.99
    assert head(z, "linear").abs().max() == 100
    with pytest.raises(ValueError):
        head(z, "unknown")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("variant", ("patch_small", "soft_small", "dynamic_small"))
def test_unit_mode_preserves_original_forward_backward_and_export_exactly(variant):
    x, t, _, warm = fixture()
    original, adapter = OriginalBank(variant), Bank(variant, "unit")
    with torch.no_grad():
        original.layers["head"].weight.fill_(0.017)
        adapter.theta.copy_(original.theta)
    xx = torch.from_numpy(x[:2]).expand(6, -1, -1)
    tt = torch.from_numpy(t[:2]).expand(6, -1, -1, -1)
    base = torch.zeros(6, 2, 3)
    a, ap = original(xx, tt, base)
    b, bp = adapter(xx, tt, base)
    assert torch.equal(a, b) and torch.equal(ap, bp)
    (a.square().sum() + ap.sum()).backward()
    (b.square().sum() + bp.sum()).backward()
    assert torch.equal(original.theta.grad, adapter.theta.grad)
    old, new = original.export(0, warm, t), adapter.export(0, warm, t)
    assert set(old) == set(new)
    for key in old:
        np.testing.assert_array_equal(old[key], new[key])
    assert not torch.cuda.is_initialized()


def test_every_architecture_can_reach_a_reference_outside_original_range():
    x, t, _, warm = fixture()
    for variant in VARIANTS:
        for mode in ("unit", "wide", "linear"):
            net = Bank(variant, mode, seeds=(17,))
            m = net.export(0, [warm[0]], t)
            np.testing.assert_allclose(
                Predictor(m)(x[:1], t[:1]), warm_predict(warm[0], x[:1]), atol=2e-8, rtol=0
            )
            # Constant residual of three normalized units: impossible with unit cap,
            # attainable exactly with the other two heads, including four passes.
            bias = 3.0 if mode != "wide" else 4 * np.arctanh(0.75)
            with torch.no_grad():
                net.layers["head"].bias.fill_(bias)
            m = net.export(0, [warm[0]], t)
            out = Predictor(m)(x[:1], t[:1])
            delta = (out - warm_predict(warm[0], x[:1])) / warm[0]["y_std"]
            if mode == "unit":
                assert np.abs(delta).max() < 1
            else:
                np.testing.assert_allclose(delta, 3.0, atol=2e-7, rtol=0)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("variant", ("patch_small", "soft_small", "dynamic_small"))
@pytest.mark.parametrize("mode", ("wide", "linear"))
def test_changed_head_has_independent_numpy_and_torch_agreement(variant, mode):
    x, t, _, warm = fixture()
    net = Bank(variant, mode, seeds=(17,))
    with torch.no_grad():
        net.layers["head"].weight.fill_(0.13)
        net.layers["head"].bias.fill_(1.3)
    m = net.export(0, [warm[0]], t)
    numpy = Predictor(m)(x[:3], t[:3], all_passes=True)
    actual = predict_torch(m, x[:3], t[:3], "cpu", all_passes=True)
    np.testing.assert_allclose(numpy, actual, atol=0.002, rtol=1e-6)
    assert not torch.cuda.is_initialized()
