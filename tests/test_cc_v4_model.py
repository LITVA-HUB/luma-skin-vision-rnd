"""SYNTHETIC CPU probes of the correction evidence core, not accuracy evidence."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    assert path.exists(), "Correction evidence model core is not implemented"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def scene(dtype=torch.float32, batch=2):
    generator = torch.Generator().manual_seed(923)
    x = torch.rand(batch, 3, 4, 4, generator=generator, dtype=dtype) * 0.3 + 0.01
    x[:, :, 0, 0] = torch.tensor([0.8, 0.05, 0.02], dtype=dtype)
    x[:, :, 3, 3] = torch.tensor([0.02, 0.03, 0.9], dtype=dtype)
    return x.repeat_interleave(16, -2).repeat_interleave(16, -1)


def test_analytic_costs_match_independent_geometry_and_joint_translation():
    model = load_script("cc_v4_model")
    oracle = load_script("cc_v4_geometry")
    hypotheses = torch.tensor([[[0.0, 0.0], [1.4, -0.9], [-1.9, 2.0]]], dtype=torch.float64)
    actions = torch.tensor([[[0.0, 0.0], [0.3, 1.2], [-3.9, 3.8]]], dtype=torch.float64)
    costs = model.analytic_costs(hypotheses, actions)
    for index in range(3):
        h = hypotheses[:, index].numpy()
        gt = np.exp(np.stack([h[:, 0], np.zeros(len(h)), h[:, 1]], axis=-1))
        expected = oracle.correction_targets(gt, actions.numpy())
        np.testing.assert_allclose(costs["angular"][:, :, index], expected["reproduction_degrees"], atol=1e-6, rtol=1e-7)
        np.testing.assert_allclose(costs["sin2"][:, :, index], expected["sin2_cost"], atol=2e-15, rtol=2e-13)
    shift = torch.tensor([0.05, -0.08], dtype=torch.float64)
    moved = model.analytic_costs(hypotheses + shift, actions + shift)
    torch.testing.assert_close(moved["sin2"], costs["sin2"], atol=1e-15, rtol=1e-13)
    torch.testing.assert_close(moved["angular"], costs["angular"], atol=1e-12, rtol=1e-12)
    assert costs["sin2"][0, 0, 0].item() == 0


def test_cost_action_gradients_match_oracle_and_finite_differences():
    model = load_script("cc_v4_model")
    oracle = load_script("cc_v4_geometry")
    h = torch.tensor([[[0.7, -0.6]]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[[0.2, 0.4], [-0.1, -0.2]]], dtype=torch.float64, requires_grad=True)
    result = model.analytic_costs(h, a)
    gradient = torch.autograd.grad(result["sin2"].sum(), a, create_graph=True)[0]
    gt = np.exp([[0.7, 0.0, -0.6]])
    expected = oracle.correction_targets(gt, a.detach().numpy())["cost_gradient"]
    np.testing.assert_allclose(gradient.detach(), expected, atol=1e-14, rtol=1e-12)
    for key in ("angular", "sin2"):
        assert torch.autograd.gradcheck(lambda q, r: model.analytic_costs(q, r)[key], (h, a), atol=1e-6, rtol=1e-5)
        assert torch.autograd.gradgradcheck(lambda q, r: model.analytic_costs(q, r)[key], (h, a), atol=2e-5, rtol=2e-4)


def test_exact_null_has_finite_second_derivatives_and_no_cost_improvement():
    model = load_script("cc_v4_model")
    h = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    a = torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True)
    costs = model.analytic_costs(h, a)
    assert costs["sin2"].abs().max() < 1e-15
    assert costs["angular"].max() < 1e-6
    for key in ("angular", "sin2"):
        gradient = torch.autograd.grad(costs[key].sum(), a, create_graph=True, retain_graph=True)[0]
        second = torch.autograd.grad(gradient.sum(), a, retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and torch.isfinite(second).all()
    net = model.CorrectionEvidenceNet().double().eval()
    cache = net.encode(scene(torch.float64, batch=1))
    cache["local_actions"] = cache["point_action"][:, None].expand(-1, 16, -1)
    selected = net.select(cache, steps=4)
    torch.testing.assert_close(selected["action"], cache["point_action"], atol=0, rtol=0)
    torch.testing.assert_close(selected["trajectory_actions"], cache["point_action"][:, None].expand(-1, 4, -1), atol=0, rtol=0)


def test_cache_shapes_proposal_prior_and_unit_positive_outputs():
    model = load_script("cc_v4_model")
    net = model.CorrectionEvidenceNet().eval()
    image = scene()
    before = image.clone()
    cache = net.encode(image)
    for key, shape in {
        "point_action": (2, 2), "local_actions": (2, 16, 2),
        "local_features": (2, 16, 48), "context": (2, 64),
        "mean_logchroma": (2, 16, 2), "rms_logchroma": (2, 16, 2), "valid": (2,),
    }.items():
        assert cache[key].shape == shape
    assert cache["valid"].all()
    assert torch.count_nonzero(cache["point_action"]) == 0
    assert torch.unique(cache["local_actions"][0], dim=0).shape[0] == 16
    assert (cache["local_actions"] - cache["point_action"][:, None]).norm(dim=-1).max() <= 0.060001
    out = net(image)
    assert out["context"].shape == (2, 64)
    assert (out["pred"] > 0).all()
    torch.testing.assert_close(out["pred"].norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(image, before, atol=0, rtol=0)


def test_local_color_is_actual_image_mean_and_rms_after_scale_safe_pooling():
    model = load_script("cc_v4_model")
    net = model.CorrectionEvidenceNet().double().eval()
    x = scene(torch.float64, batch=1)
    x[:, :, :16, :16] *= torch.linspace(0.2, 1.8, 16, dtype=x.dtype)[None, None, None, :]
    cache = net.encode(x)
    tile = x[0, :, :16, :16]
    mean = tile.mean((-2, -1)).numpy()
    rms = tile.square().mean((-2, -1)).sqrt().numpy()
    np.testing.assert_allclose(cache["mean_logchroma"][0, 0].detach(), np.log(mean[[0, 2]] / mean[1]), atol=1e-12)
    np.testing.assert_allclose(cache["rms_logchroma"][0, 0].detach(), np.log(rms[[0, 2]] / rms[1]), atol=1e-12)


def test_queries_normalize_weights_and_transport_observed_color_per_action():
    model = load_script("cc_v4_model")
    torch.manual_seed(235)
    net = model.CorrectionEvidenceNet().double().eval()
    cache = net.encode(scene(torch.float64))
    actions = torch.tensor([[[0.0, 0.0], [0.8, -0.7]]], dtype=torch.float64).expand(2, -1, -1)
    captured = []
    handle = net.router[0].register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
    transport = net.query(cache, actions)
    handle.remove()
    router_input = captured[0].reshape(2, 2, 16, 118)
    for field, start in (("mean_logchroma", 112), ("rms_logchroma", 114)):
        corrected = (cache[field][:, None] - actions[:, :, None]).detach().numpy()
        rgb = np.exp(np.stack((corrected[..., 0], np.zeros(corrected.shape[:-1]), corrected[..., 1]), -1))
        expected = 3 * rgb[..., [0, 2]] / rgb.sum(-1, keepdims=True) - 1
        np.testing.assert_allclose(router_input[..., start:start + 2].detach(), expected, atol=1e-14)
    torch.testing.assert_close(router_input[..., 116:118], (actions - cache["point_action"][:, None])[:, :, None].expand(-1, -1, 16, -1))
    assert (transport["weights"][:, 0] - transport["weights"][:, 1]).abs().max() > 1e-7
    net.mode = "posterior"
    posterior = net.query(cache, actions)
    torch.testing.assert_close(posterior["weights"][:, 0], posterior["weights"][:, 1], atol=0, rtol=0)
    for output in (transport, posterior):
        torch.testing.assert_close(output["weights"].sum(-1), torch.ones(2, 2, dtype=torch.float64))
        assert ((output["angular_risk"] >= 0) & (output["angular_risk"] < 55)).all()
        assert ((output["sin2_risk"] >= 0) & (output["sin2_risk"] <= 2 / 3)).all()
    for i in range(2):
        single = net.query(cache, actions[:, i:i + 1])
        for key in posterior:
            torch.testing.assert_close(single[key][:, 0], posterior[key][:, i])


def test_action_null_keeps_centered_color_but_responds_to_relative_action():
    model = load_script("cc_v4_model")
    torch.manual_seed(426)
    net = model.CorrectionEvidenceNet(mode="action").double().eval()
    with torch.no_grad():
        net.point_head.bias.copy_(torch.tensor([0.2, -0.15], dtype=torch.float64))
    cache = net.encode(scene(torch.float64))
    actions = torch.tensor([[[0.0, 0.0], [0.8, -0.7]]], dtype=torch.float64).expand(2, -1, -1)
    captured = []
    handle = net.router[0].register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
    action_output = net.query(cache, actions)
    handle.remove()
    router_input = captured[0].reshape(2, 2, 16, 118)
    torch.testing.assert_close(router_input[:, 0, :, 112:116], router_input[:, 1, :, 112:116], atol=0, rtol=0)
    torch.testing.assert_close(router_input[..., 116:118], (actions - cache["point_action"][:, None])[:, :, None].expand(-1, -1, 16, -1))
    assert (action_output["weights"][:, 0] - action_output["weights"][:, 1]).abs().max() > 1e-7
    net.mode = "posterior"
    posterior = net.query(cache, actions)
    torch.testing.assert_close(posterior["weights"][:, 0], posterior["weights"][:, 1], atol=0, rtol=0)
    net.mode = "transport"
    transport = net.query(cache, actions)
    assert (transport["weights"] - action_output["weights"]).abs().max() > 1e-7


def test_corrected_simplex_is_physical_nonlinear_transport_with_second_derivatives():
    model = load_script("cc_v4_model")
    color = torch.tensor([[math.log(2), math.log(0.5)]], dtype=torch.float64, requires_grad=True)
    action = torch.tensor([[math.log(2), math.log(0.5)]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(model.corrected_simplex(color, action), torch.zeros(1, 2, dtype=torch.float64), atol=1e-15, rtol=0)
    assert torch.autograd.gradcheck(model.corrected_simplex, (color, action), atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(model.corrected_simplex, (color, action), atol=1e-7, rtol=1e-5)
    left = model.corrected_simplex(color, torch.tensor([[-1.0, 0.0]], dtype=torch.float64))
    right = model.corrected_simplex(color, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    middle = model.corrected_simplex(color, torch.zeros(1, 2, dtype=torch.float64))
    assert (middle - (left + right) / 2).abs().max() > 0.01


def test_sobolev_loss_backpropagates_through_vote_router_and_backbone():
    model = load_script("cc_v4_model")
    torch.manual_seed(37)
    net = model.CorrectionEvidenceNet().train()
    cache = net.encode(scene())
    actions = torch.tensor([[[0.15, -0.2], [-0.25, 0.35]]], requires_grad=True).expand(2, -1, -1)
    query = net.query(cache, actions)
    grad = torch.autograd.grad(query["sin2_risk"].sum(), actions, create_graph=True)[0]
    loss = query["angular_risk"].mean() + 5 * (grad - 0.13).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    for component in (net.vote_head, net.router, net.backbone):
        gradients = [p.grad for p in component.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().max() > 0 for g in gradients)


def test_action_gradient_penalty_alone_reaches_all_evidence_components():
    model = load_script("cc_v4_model")
    torch.manual_seed(81)
    net = model.CorrectionEvidenceNet().train()
    cache = net.encode(scene())
    actions = torch.full((2, 2, 2), 0.27, requires_grad=True)
    query = net.query(cache, actions)
    gradient = torch.autograd.grad(query["sin2_risk"].sum(), actions, create_graph=True)[0]
    penalty = (gradient - 0.13).square().mean()
    penalty.backward()
    for component in (net.vote_head, net.router, net.backbone):
        gradients = [p.grad for p in component.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().max() > 0 for g in gradients)


def test_minimum_size_singleton_deployment_works_and_training_guidance_is_explicit():
    model = load_script("cc_v4_model")
    net = model.CorrectionEvidenceNet().eval()
    image = torch.ones(1, 3, 32, 32)
    with torch.no_grad():
        out = net(image)
    assert out["valid"].item()
    assert torch.isfinite(out["pred"]).all()
    net.train()
    with pytest.raises(ValueError, match=r"eval|batch"):
        net(image)


@pytest.mark.parametrize("steps,queries", [(1, 25), (2, 51), (4, 103)])
@pytest.mark.parametrize("mode", ["transport", "action", "posterior"])
def test_deterministic_selection_has_declared_budget_and_keeps_previous_and_base(steps, queries, mode):
    model = load_script("cc_v4_model")
    torch.manual_seed(74)
    net = model.CorrectionEvidenceNet(mode=mode).eval()
    with torch.no_grad():
        net.vote_head[-1].bias.copy_(torch.tensor([0.4, -0.3]))
        cache = net.encode(scene())
        result = net.select(cache, steps=steps)
        again = net.select(cache, steps=steps)
    assert result["query_count"] == queries
    assert result["trajectory_actions"].shape == (2, steps, 2)
    assert result["trajectory_risk"].shape == (2, steps)
    torch.testing.assert_close(result["action"], result["trajectory_actions"][:, -1])
    torch.testing.assert_close(result["risk"], result["trajectory_risk"][:, -1])
    assert (result["trajectory_risk"][:, 1:] <= result["trajectory_risk"][:, :-1] + 1e-6).all()
    assert (result["action"].abs() <= 2).all()
    torch.testing.assert_close(result["action"], again["action"], atol=0, rtol=0)
    radii = [0.24, 0.06, 0.03, 0.015]
    point = cache["point_action"]
    previous = point
    axis = torch.linspace(-1, 1, 5)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack((xx.flatten(), yy.flatten()), -1)
    for stage in range(steps):
        candidates = (previous[:, None] + radii[stage] * offsets).clamp(-2, 2)
        if stage:
            candidates = torch.cat((candidates, point[:, None]), 1)
        risk = net.query(cache, candidates)["angular_risk"]
        chosen = result["trajectory_actions"][:, stage]
        assert (candidates == chosen[:, None]).all(-1).any(-1).all()
        torch.testing.assert_close(result["trajectory_risk"][:, stage], risk.min(-1).values)
        previous = chosen


@pytest.mark.parametrize("kind", ["black", "nan", "infinity", "negative", "absent_red", "absent_green", "absent_blue"])
def test_invalid_images_never_accepted_and_have_finite_diagnostics(kind):
    model = load_script("cc_v4_model")
    net = model.CorrectionEvidenceNet().eval()
    x = scene()
    if kind == "black":
        x[0] = 0
    elif kind.startswith("absent"):
        x[0, {"absent_red": 0, "absent_green": 1, "absent_blue": 2}[kind]] = 0
    else:
        x[0, 0, 0, 0] = {"nan": float("nan"), "infinity": float("inf"), "negative": -0.1}[kind]
    cache = net.encode(x)
    out = net.select(cache)
    assert out["valid"].tolist() == [False, True]
    for value in list(cache.values()) + list(out.values()):
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
    assert (out["pred"] > 0).all()
    torch.testing.assert_close(out["pred"].norm(dim=-1), torch.ones(2))


def test_exposure_invariance_and_finite_empty_cells_at_extreme_scales():
    model = load_script("cc_v4_model")
    net = model.CorrectionEvidenceNet().double().eval()
    x = scene(torch.float64)
    x[:, :, 16:32, 16:32] = 0
    base = net.encode(x)
    for scale in (1e-200, 1e200):
        scaled = net.encode(x * scale)
        assert scaled["valid"].all()
        for key in ("context", "local_features", "point_action", "local_actions", "mean_logchroma", "rms_logchroma"):
            assert torch.isfinite(scaled[key]).all()
            torch.testing.assert_close(scaled[key], base[key], atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("bad", [torch.zeros(0, 3, 32, 32), torch.zeros(1, 4, 32, 32), torch.zeros(1, 3, 31, 32), torch.zeros(1, 3, 32, 32, dtype=torch.int64)])
def test_structurally_invalid_images_raise(bad):
    model = load_script("cc_v4_model")
    with pytest.raises(ValueError):
        model.CorrectionEvidenceNet().encode(bad)


@pytest.mark.parametrize("bad", [torch.zeros(2, 2), torch.zeros(2, 0, 2), torch.zeros(1, 3, 2), torch.full((2, 1, 2), 4.01), torch.full((2, 1, 2), float("nan"))])
def test_query_enforces_finite_bounded_action_domain(bad):
    model = load_script("cc_v4_model")
    net = model.CorrectionEvidenceNet().eval()
    cache = net.encode(scene())
    with pytest.raises((ValueError, RuntimeError)):
        net.query(cache, bad)


def test_half_inputs_and_autocast_still_compute_fp32_and_double_is_supported():
    model = load_script("cc_v4_model")
    net = model.CorrectionEvidenceNet().eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = net(scene().half())
    assert output["pred"].dtype == torch.float32
    assert output["context"].dtype == torch.float32
    assert net.double()(scene(torch.float64))["pred"].dtype == torch.float64


def test_mode_parameter_graph_matches_and_direct_uses_only_point_at_inference():
    model = load_script("cc_v4_model")
    nets = [model.CorrectionEvidenceNet(mode=mode).eval() for mode in ("transport", "action", "posterior", "direct")]
    graphs = [{name: tuple(p.shape) for name, p in net.named_parameters()} for net in nets]
    assert graphs[0] == graphs[1] == graphs[2] == graphs[3]
    count = sum(p.numel() for p in nets[0].parameters())
    assert 1_000_000 <= count <= 5_000_000
    direct = nets[-1]
    with torch.no_grad():
        direct.point_head.bias.copy_(torch.tensor([0.8, -0.6]))
    cache = direct.encode(scene())
    for steps in (1, 2, 4):
        result = direct.select(cache, steps=steps)
        torch.testing.assert_close(result["action"], cache["point_action"])
        assert result["query_count"] == 1
        assert result["trajectory_actions"].shape == (2, 1, 2)
        torch.testing.assert_close(result["pred"], result["base_pred"])
    with pytest.raises(ValueError):
        nets[0].select(cache, steps=3)
    with pytest.raises(ValueError):
        model.CorrectionEvidenceNet(mode="unknown")
