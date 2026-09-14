"""Behavioral numerical checks for the new architecture scaling experiment."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_architecture_scale import VARIANTS, Bank, capacity, fit, predict
from chromaseed_neural_prefix_numpy import predict as warm_predict


def fixture():
    r = np.random.default_rng(778)
    x = r.normal(size=(23, 36)).astype(np.float32)
    t = r.normal(size=(23, 64, 18)).astype(np.float32)
    y = r.normal(size=(23, 3)) * 6 + [52, 14, 19]
    warm = []
    for _ in range(3):
        warm.append(
            dict(
                family=np.asarray("blind4"),
                original_k=np.asarray(4, np.uint8),
                prefix=np.asarray(2, np.uint8),
                x_mean=x.astype(float).mean(0).astype(np.float32),
                x_std=x.astype(float).std(0).astype(np.float32),
                y_mean=y.mean(0).astype(np.float32),
                y_std=y.std(0).astype(np.float32),
                w0=r.normal(0, 0.03, (36, 16)).astype(np.float32),
                b0=np.zeros(16, np.float32),
                v0=r.normal(0, 0.03, (16, 3)).astype(np.float32),
                c0=np.zeros(3, np.float32),
            )
        )
    return x, t, y, warm


def test_every_architecture_starts_at_same_warm_function_and_has_real_capacity():
    x, t, _, warm = fixture()
    for variant in VARIANTS:
        bank = Bank(variant)
        assert bank.theta.shape == (6, capacity(variant) - 643)
        m = bank.export(0, warm, t)
        np.testing.assert_allclose(predict(m, x, t), warm_predict(warm[0], x), atol=2e-8, rtol=0)
        if variant.endswith("5m"):
            assert 4_800_000 < capacity(variant) < 5_000_000


def test_training_uses_tokens_and_export_matches_torch():
    x, t, y, warm = fixture()
    for variant in ("patch_small", "soft_small", "dynamic_small"):
        models, info = fit(x, t, y, np.ones(len(x)), warm, variant, 4, (4,), "cpu", "eager")
        assert info["steps"] == 4
        model = models[4][0]
        out = predict(model, x, t)
        assert np.max(np.abs(out - warm_predict(warm[0], x))) > 1e-5
        assert np.max(np.abs(out - predict(model, x, np.zeros_like(t)))) > 1e-6
        from chromaseed_architecture_scale import predict_torch

        np.testing.assert_allclose(out, predict_torch(model, x, t, "cpu"), atol=2e-4, rtol=1e-6)


def test_bank_slot_isolation_and_dynamic_gate_gradient():
    from chromaseed_architecture_scale import gate

    scores = torch.tensor([[[-2.0, -1.0, -0.1, 0.2, 2.0, 3.0]]], requires_grad=True)
    a, count, penalty = gate(scores, "dynamic", True)
    assert count.item() == 4
    torch.testing.assert_close(a.sum(-1), torch.ones(1, 1))
    (a[..., 0].sum() + penalty.sum()).backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0
    x, t, _, _ = fixture()
    bank = Bank("soft_small")
    xx = torch.from_numpy(x[:2]).expand(6, -1, -1)
    tt = torch.from_numpy(t[:2]).expand(6, -1, -1, -1)
    base = torch.zeros(6, 2, 3)
    out, _ = bank(xx, tt, base)
    out[0].square().sum().add(out[0].sum()).backward()
    assert bank.theta.grad[0].abs().sum() > 0
    assert bank.theta.grad[1:].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_largest_cuda_replay_and_prefix_are_exact():
    x, t, y, warm = fixture()
    args = (x, t, y, np.ones(len(x)), warm, "dynamic5m")
    a, _ = fit(*args, 8, (4, 8), "cuda", "cuda_graph")
    b, _ = fit(*args, 8, (4, 8), "cuda", "cuda_graph")
    c, _ = fit(*args, 4, (4,), "cuda", "cuda_graph")
    for step in (4, 8):
        for slot in range(6):
            np.testing.assert_array_equal(a[step][slot]["theta"], b[step][slot]["theta"])
    for slot in range(6):
        np.testing.assert_array_equal(a[4][slot]["theta"], c[4][slot]["theta"])
    from chromaseed_architecture_scale import predict_torch

    np.testing.assert_allclose(
        predict(a[8][0], x, t), predict_torch(a[8][0], x, t, "cuda"), atol=0.002, rtol=1e-6
    )
