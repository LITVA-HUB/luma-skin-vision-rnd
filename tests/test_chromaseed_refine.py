import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_bank_adamw_matches_independent_torch_optimizers_with_clipping():
    from chromaseed_refine import BankAdamW

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(531)
    params = [torch.nn.Parameter(torch.randn(3, 2, 4, generator=generator, dtype=torch.float64)),
              torch.nn.Parameter(torch.randn(3, 4, generator=generator, dtype=torch.float64))]
    rates = [.0003, .001, .003]
    reference = [[torch.nn.Parameter(p[s].detach().clone()) for p in params] for s in range(3)]
    optimizers = [torch.optim.AdamW(p, lr=lr, weight_decay=.01, eps=1e-8)
                  for p, lr in zip(reference, rates, strict=True)]
    bank = BankAdamW(params, rates, weight_decay=.01, max_norm=5.)
    for _ in range(13):
        for p in params:
            p.grad = 9 * torch.randn(p.shape, generator=generator, dtype=p.dtype)
        for s, optimizer in enumerate(optimizers):
            for p, q in zip(params, reference[s], strict=True):
                q.grad = p.grad[s].clone()
            torch.nn.utils.clip_grad_norm_(reference[s], 5.)
            optimizer.step()
        bank.step()
        for s in range(3):
            for p, q in zip(params, reference[s], strict=True):
                torch.testing.assert_close(p[s], q, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("family", ["stats_mlp", "patch_mlp", "recur_soft", "recur_top16", "recur_dynamic"])
def test_bank_network_is_independent_and_exportable(family):
    from chromaseed_refine import BankAdamW, BankNet

    torch.set_num_threads(1)
    net = BankNet(family, [17, 29, 17]).double()
    single = BankNet(family, [17]).double()
    generator = torch.Generator().manual_seed(813)
    x = torch.randn(1, 3, 36, generator=generator, dtype=torch.float64)
    patches = torch.randn(1, 3, 64, 18, generator=generator, dtype=torch.float64)
    base = torch.randn(1, 3, 3, generator=generator, dtype=torch.float64)
    target = torch.randn(1, 3, 3, generator=generator, dtype=torch.float64)
    bank_optimizer = BankAdamW(net.parameters(), [.001, .002, .001])
    single_optimizer = torch.optim.AdamW(single.parameters(), lr=.001, weight_decay=.01)
    for _ in range(3):
        net.zero_grad(set_to_none=True)
        single.zero_grad(set_to_none=True)
        predictions, _, penalty = net(x.expand(3, -1, -1), patches.expand(3, -1, -1, -1), base.expand(3, -1, -1))
        pred_single, _, penalty_single = single(x, patches, base)
        torch.testing.assert_close(predictions[0], pred_single[0], rtol=1e-8, atol=1e-9)
        loss = (predictions - target[:, :, None]).square().mean((1, 2, 3)) + .001 * penalty
        loss_single = (pred_single - target[:, :, None]).square().mean() + .001 * penalty_single.sum()
        loss.sum().backward()
        loss_single.backward()
        bank_optimizer.step()
        torch.nn.utils.clip_grad_norm_(single.parameters(), 5.)
        single_optimizer.step()
    for name, p in net.named_parameters():
        torch.testing.assert_close(p[0], dict(single.named_parameters())[name][0], rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(p[0], p[2], rtol=1e-10, atol=1e-11)
    exported = net.export_slot(0)
    restored = BankNet.from_payload(exported).double()
    actual = restored(x, patches, base)[0]
    expected = net(x.expand(3, -1, -1), patches.expand(3, -1, -1, -1), base.expand(3, -1, -1))[0][:1]
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_dynamic_gate_changes_connection_count_and_retains_learning_signal():
    from chromaseed_refine import gate_weights

    scores = torch.tensor([[[-2.] * 20, [-2.] * 10 + [1.] * 10, [1.] * 20]], requires_grad=True)
    weights, count, penalty = gate_weights(scores, "recur_dynamic", training=True)
    torch.testing.assert_close(count, torch.tensor([[4., 10., 20.]]))
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 3))
    (weights.mul(torch.arange(20)).sum() + penalty.sum()).backward()
    assert torch.isfinite(scores.grad).all() and torch.count_nonzero(scores.grad) == 60
    _, fixed_count, _ = gate_weights(scores.detach(), "recur_top16", training=False)
    torch.testing.assert_close(fixed_count, torch.full((1, 3), 16.))
    _, soft_count, _ = gate_weights(scores.detach(), "recur_soft", training=False)
    torch.testing.assert_close(soft_count, torch.full((1, 3), 20.))


def test_adaptive_exit_really_stops_and_matches_full_trace_policy():
    from chromaseed_refine import BankNet, apply_exit_policy, predict_one

    net = BankNet("recur_dynamic", [17]).eval()
    with torch.no_grad():
        net.layers["head"].bias.fill_(.1)
    x, patches, base = torch.zeros(1, 1, 36), torch.zeros(1, 1, 64, 18), torch.zeros(1, 1, 3)
    scale = np.array([25., 30., 30.], np.float32)
    with torch.no_grad():
        outputs = net(x, patches, base)[0][0].numpy() * scale
    for threshold, expected_steps in ((0., 4), (100., 2)):
        selected, steps = apply_exit_policy(outputs, threshold)
        actual, executed, counts = predict_one(net, x, patches, base, scale, threshold)
        assert executed == expected_steps == steps[0] and len(counts) == executed
        np.testing.assert_allclose(actual.numpy()[0, 0] * scale, selected[0], rtol=1e-6, atol=1e-6)


def test_fold_preprocessor_does_not_depend_on_held_rows_and_anchor_learns():
    from chromaseed_refine import fit_preprocessor, transform

    rng = np.random.default_rng(168)
    x = rng.normal(size=(96, 36)).astype(np.float32)
    patches = rng.normal(size=(96, 64, 18)).astype(np.float32)
    y = np.column_stack((50 + 10 * x[:, 0], 4 + 3 * x[:, 1], 8 + 2 * x[:, 2])).astype(np.float32)
    fit = np.arange(72)
    prep = fit_preprocessor(x[fit], patches[fit], y[fit], np.ones(len(fit)))
    _, _, anchor = transform(prep, x, patches)
    predicted = anchor * prep["y_std"] + prep["y_mean"]
    assert np.mean((predicted[72:] - y[72:]) ** 2) < .2
    x[72:] = 1000
    patches[72:] = 1000
    y[72:] = 1000
    again = fit_preprocessor(x[fit], patches[fit], y[fit], np.ones(len(fit)))
    for key in prep:
        np.testing.assert_array_equal(prep[key], again[key])
    with pytest.raises(ValueError):
        fit_preprocessor(x[:0], patches[:0], y[:0], np.ones(0))
