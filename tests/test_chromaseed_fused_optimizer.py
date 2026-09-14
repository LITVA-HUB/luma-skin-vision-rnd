"""CPU-only numerical behavior; this suite never initializes a CUDA context."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_fused_optimizer import FusedBankAdamW
from chromaseed_refine import BankAdamW


def test_slot_rates_clip_decay_and_moments_against_float64_equations():
    rng = np.random.default_rng(4921)
    initial = rng.normal(size=(6, 31)).astype(np.float32)
    theta = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    rates = np.array([1e-5, 1e-4, 3e-5, 3e-4, 1e-3, 1e-2], np.float32)
    optimizer = FusedBankAdamW(theta, rates)
    expected, m, v = initial.astype(float), np.zeros((6, 31)), np.zeros((6, 31))
    original_pointer = theta.data_ptr()
    for step in range(1, 33):
        gradient = rng.normal(size=(6, 31)).astype(np.float32)
        gradient[0] = 0
        gradient[1] *= 100
        actual_rate = (rates * np.float32(0.2 + 0.8 * step / 32)).astype(np.float32)
        theta.grad = torch.from_numpy(gradient.copy())
        optimizer.lrs.copy_(torch.from_numpy(actual_rate))
        optimizer.step()
        grad = gradient.astype(float)
        grad *= np.minimum(5 / (np.linalg.norm(grad, axis=1) + 1e-6), 1)[:, None]
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        expected *= 1 - actual_rate[:, None] * 0.01
        expected -= (
            actual_rate[:, None] * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        )
        np.testing.assert_allclose(theta.detach().numpy(), expected, atol=3e-6, rtol=3e-6)
        np.testing.assert_allclose(optimizer.m.numpy(), m, atol=3e-7, rtol=3e-6)
        np.testing.assert_allclose(optimizer.v.numpy(), v, atol=3e-8, rtol=3e-6)
        assert theta.data_ptr() == original_pointer
    assert not torch.cuda.is_initialized()


def test_reset_replays_exactly_without_replacing_parameter_storage():
    theta = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 102).reshape(6, 17))
    initial = theta.detach().clone()
    optimizer = FusedBankAdamW(theta, [1e-5, 1e-4] * 3)
    original_pointer = theta.data_ptr()
    outputs = []
    for _ in range(2):
        optimizer.reset(initial)
        for step in range(1, 12):
            theta.grad = torch.cos(theta.detach() * step)
            optimizer.step()
        outputs.append(
            (
                theta.detach().clone(),
                optimizer.m.clone(),
                optimizer.v.clone(),
                optimizer.steps.clone(),
            )
        )
    for a, b in zip(*outputs, strict=True):
        assert torch.equal(a, b)
    assert theta.data_ptr() == original_pointer
    assert not torch.cuda.is_initialized()


def test_matches_existing_optimizer_numerically_for_identical_gradient_stream():
    rng = np.random.default_rng(7712)
    initial = rng.normal(size=(6, 67)).astype(np.float32)
    old = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    new = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    rates = np.array([1e-5, 1e-4] * 3, np.float32)
    reference = BankAdamW([old], rates.copy())
    candidate = FusedBankAdamW(new, rates.copy())
    for step in range(1, 65):
        g = torch.from_numpy(rng.normal(size=initial.shape).astype(np.float32))
        old.grad, new.grad = g.clone(), g.clone()
        reference.step()
        candidate.step()
        torch.testing.assert_close(old, new, atol=2e-6, rtol=2e-6)
    assert not torch.cuda.is_initialized()


def test_actual_four_pass_network_with_coupled_gradients():
    from chromaseed_fused_optimizer_probe import coupled_probe

    result = coupled_probe(16)
    assert result["max_normalized_output_abs"] <= 1e-5
    assert not torch.cuda.is_initialized()
