"""Behavioral ND tests: local credit assignment and target-free refinement."""

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def module(name):
    assert importlib.util.find_spec(name) is not None, f"missing ND implementation: {name}"
    return importlib.import_module(name)


def fixture_data():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(32, 36)).astype(np.float32)
    y = np.column_stack((50 + 3 * x[:, 0], 15 + 2 * x[:, 1], 20 + x[:, 2]))
    return x, y.astype(np.float64), np.ones(32)


def test_local_loss_cannot_change_another_block():
    m = module("chromaseed_local_denoise")
    net = m.Bank("local4", [17])
    x = torch.ones(1, 7, 36)
    state = torch.ones(1, 4, 7, 3)
    net.local(x, state)[:, 2].square().sum().backward()
    gradient = net.theta.grad.reshape(1, 4, -1)
    assert torch.count_nonzero(gradient[:, 2]) > 0
    assert torch.count_nonzero(gradient[:, [0, 1, 3]]) == 0


def test_e2e_final_loss_reaches_earlier_blocks():
    m = module("chromaseed_local_denoise")
    net = m.Bank("e2e4", [17])
    net.rollout(torch.ones(1, 7, 36))[:, -1].square().sum().backward()
    assert torch.count_nonzero(net.theta.grad[0]) > 0


def test_bank_reordering_preserves_each_network():
    m = module("chromaseed_local_denoise")
    x = torch.ones(2, 5, 36)
    actual = m.Bank("local2", [29, 17]).rollout(x).detach().numpy()
    for slot, seed in enumerate((29, 17)):
        expected = m.Bank("local2", [seed]).rollout(x[:1]).detach().numpy()[0]
        np.testing.assert_array_equal(actual[slot], expected)


def test_two_stage_consumer_receives_latent_and_returns_clean_color():
    m = module("chromaseed_local_denoise")
    consumer = module("chromaseed_local_denoise_numpy")
    net = m.Bank("local2", [17])
    with torch.no_grad():
        net.theta.zero_()
        net.theta[0, -3:] = torch.tensor([1.0, 2.0, 3.0])
        # Block2 is identity on positive state coordinates: state -> ReLU -> output.
        for k in range(3):
            net.theta[1, (36 + k) * 32 + k] = 1
            net.theta[1, 39 * 32 + 32 + k * 3 + k] = 1
    payload = dict(
        family=np.asarray("local2"),
        theta=net.theta.detach().numpy().copy(),
        x_mean=np.zeros(36, np.float32),
        x_std=np.ones(36, np.float32),
        y_mean=np.zeros(3, np.float32),
        y_std=np.ones(3, np.float32),
    )
    predictor = consumer.Predictor(payload)
    expected = np.array([[1.0, 2.0, 3.0], [2**-0.5, 2**0.5, 3 * 2**-0.5]])
    np.testing.assert_allclose(predictor.trace(np.zeros(36)), expected, atol=1e-7)
    np.testing.assert_allclose(predictor(np.zeros(36)), expected[-1], atol=1e-7)


@pytest.mark.parametrize("bad", [np.zeros((1, 36)), np.zeros(35), np.full(36, np.nan)])
def test_actual_consumer_rejects_malformed_input(bad):
    m = module("chromaseed_local_denoise")
    consumer = module("chromaseed_local_denoise_numpy")
    x, y, _ = fixture_data()
    payload = m.Bank("plain", [17]).export(0, m.preprocessor(x, y))
    with pytest.raises(ValueError):
        consumer.Predictor(payload)(bad)


def test_blind_model_never_uses_state_coordinates():
    m = module("chromaseed_local_denoise")
    net = m.Bank("blind4", [17])
    x = torch.ones(1, 3, 36)
    a = net.local(x, torch.zeros(1, 4, 3, 3))
    b = net.local(x, torch.full((1, 4, 3, 3), 900.0))
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_preprocessor_only_depends_on_supplied_fit_rows():
    m = module("chromaseed_local_denoise")
    x, y, _ = fixture_data()
    prep = m.preprocessor(x[:2], y[:2])
    np.testing.assert_allclose(prep["x_mean"], x[:2].astype(float).mean(0), rtol=1e-6)
    np.testing.assert_allclose(prep["y_mean"], y[:2].mean(0), rtol=1e-6)
    with pytest.raises(ValueError):
        m.preprocessor(x, y[:1])


def test_training_bank_matches_separate_cpu_models():
    fitter = module("chromaseed_local_denoise_fit")
    x, y, w = fixture_data()
    slots = [(17, 0.001), (29, 0.003)]
    packed, _ = fitter.fit(x, y, w, "local2", slots, 8, (8,), "cpu", "eager")
    for slot in range(2):
        single, _ = fitter.fit(x, y, w, "local2", [slots[slot]], 8, (8,), "cpu", "eager")
        np.testing.assert_allclose(
            packed[8][slot]["theta"], single[8][0]["theta"], rtol=1e-6, atol=1e-7
        )


def test_cuda_graph_warmup_is_fully_reset():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    fitter = module("chromaseed_local_denoise_fit")
    x, y, w = fixture_data()
    for family in ("local4", "e2e4"):
        a, _ = fitter.fit(x, y, w, family, [(17, 0.001)], 8, (8,), "cuda", "eager")
        b, _ = fitter.fit(x, y, w, family, [(17, 0.001)], 8, (8,), "cuda", "cuda_graph")
        np.testing.assert_allclose(a[8][0]["theta"], b[8][0]["theta"], rtol=2e-6, atol=2e-6)


def test_noise_is_independent_of_slot_order_and_learning_rate():
    fitter = module("chromaseed_local_denoise_fit")
    a = fitter.noise_sequences([17, 29, 17], 4, 4)
    b = fitter.noise_sequences([29], 4, 4)
    np.testing.assert_array_equal(a[0], a[2])
    np.testing.assert_array_equal(a[1], b[0])
    assert not np.array_equal(a[0], a[1])
    assert not np.array_equal(a[0, :, 0], a[0, :, 1])


def test_per_block_optimizer_matches_individual_adamw_with_clipping():
    optimizer_module = module("chromaseed_refine")
    initial = torch.arange(28, dtype=torch.float32).reshape(4, 7) / 31
    packed = torch.nn.Parameter(initial.clone())
    rates = [0.001, 0.001, 0.003, 0.003]
    opt = optimizer_module.BankAdamW([packed], rates)
    separate = [torch.nn.Parameter(row.clone()) for row in initial]
    refs = [
        torch.optim.AdamW([p], lr=lr, weight_decay=0.01, foreach=False)
        for p, lr in zip(separate, rates)
    ]
    for step in range(3):
        gradient = torch.arange(28, dtype=torch.float32).reshape(4, 7) * (step + 1) / 9
        packed.grad = gradient.clone()
        opt.step()
        for index, (p, o) in enumerate(zip(separate, refs)):
            p.grad = gradient[index].clone()
            torch.nn.utils.clip_grad_norm_([p], 5.0)
            o.step()
    torch.testing.assert_close(packed, torch.stack(separate), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("family", ["plain", "local2", "local4", "blind4", "e2e4"])
def test_export_all_stages_matches_float32_training_path(family):
    m = module("chromaseed_local_denoise")
    consumer = module("chromaseed_local_denoise_numpy")
    x, y, _ = fixture_data()
    prep = m.preprocessor(x, y)
    net = m.Bank(family, [17])
    payload = net.export(0, prep)
    xn = torch.from_numpy((x - prep["x_mean"]) / prep["x_std"])[None]
    expected = (
        net.rollout(xn).detach().numpy()[0].transpose(1, 0, 2) * prep["y_std"] + prep["y_mean"]
    )
    actual = consumer.predict(payload, x)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.002)
    single = consumer.Predictor(payload)
    np.testing.assert_allclose(
        np.stack([single(row) for row in x]), actual[:, -1], rtol=0, atol=2e-8
    )
