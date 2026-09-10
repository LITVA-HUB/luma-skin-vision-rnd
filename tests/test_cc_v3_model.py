"""SYNTHETIC CPU numerical probes, never benchmark accuracy evidence."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F


def core():
    path = Path(__file__).resolve().parents[1] / "scripts/cc_v3_model.py"
    assert path.exists(), "Task1 model core is not implemented"
    spec = importlib.util.spec_from_file_location("cc_v3_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def scene(dtype=torch.float64):
    generator = torch.Generator().manual_seed(341)
    patches = torch.rand(2, 3, 4, 4, generator=generator, dtype=dtype) * 0.2 + 0.05
    patches[:, :, 0, 0] = torch.tensor([1.0, 0.08, 0.09], dtype=dtype)
    patches[:, :, 0, 1] = torch.tensor([0.07, 1.1, 0.06], dtype=dtype)
    patches[:, :, 0, 2] = torch.tensor([0.09, 0.06, 0.95], dtype=dtype)
    return patches.repeat_interleave(4, -2).repeat_interleave(4, -1)


def test_frame_selection_is_max_volume_and_lexicographic_on_ties():
    m = core()
    x = scene()
    net = m.ColorFramePosteriorNet().double()
    out = net(x)
    assert out["valid"].all()
    assert out["frame_indices"].tolist() == [[0, 1, 2], [0, 1, 2]]
    x[:, :, 0:4, 12:16] = x[:, :, 0:4, 0:4]
    tie = net(x)
    assert tie["frame_indices"].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert (tie["frame_volume_margin"] == 0).all()


@pytest.mark.parametrize("transform", [
    [[1.2, 0.12, 0.04], [0.04, 0.9, 0.09], [0.1, 0.06, 1.1]],
    [[1.3, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 1.1]],
])
def test_nontrivial_learned_point_and_posterior_are_projectively_equivariant(transform):
    m = core()
    torch.manual_seed(27)
    net = m.ColorFramePosteriorNet().double()
    with torch.no_grad():
        net.point_head.weight.normal_(0, 0.001)
        net.point_head.bias.copy_(torch.tensor([0.02, -0.01, 0.03]))
    x = scene()
    matrix = torch.tensor(transform, dtype=x.dtype)
    out = net(x)
    mixed = net(torch.einsum("ij,njhw->nihw", matrix, x))
    assert out["valid"].all() and mixed["valid"].all()
    torch.testing.assert_close(mixed["pred"], F.normalize(out["pred"] @ matrix.T, dim=-1), atol=2e-9, rtol=2e-9)
    for key in ["context", "directions", "logits", "concentration", "canonical_mean"]:
        torch.testing.assert_close(mixed[key], out[key], atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("mode", ["direct", "diagonal", "frame"])
def test_forward_loss_and_actual_backward_are_finite(mode):
    m = core()
    net = m.ColorFramePosteriorNet(mode=mode)
    out = net(scene(torch.float32))
    assert out["context"].shape == (2, 64)
    assert out["directions"].shape == (2, 8, 3)
    assert out["frame"].dtype == torch.float64
    assert out["valid"].all()
    assert (out["pred"] > 0).all()
    for value in out.values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
    loss = m.posterior_nll(out, torch.tensor([[0.8, 1., 0.9], [1., .7, .9]]))
    loss = loss + (out["pred"] - .5).square().mean() + out["transport_risk"].mean() * 0.001
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_modes_have_identical_trainable_parameter_shapes_and_count():
    m = core()
    signatures = [{k: tuple(p.shape) for k, p in m.ColorFramePosteriorNet(mode=mode).named_parameters()} for mode in ["direct", "diagonal", "frame"]]
    assert signatures[0] == signatures[1] == signatures[2]


def test_invalid_input_and_rank_deficiency_refuse_with_finite_positive_diagnostics():
    m = core()
    x = scene(torch.float32)[:1].repeat(6, 1, 1, 1)
    x[0] = 0
    x[1] = 0.5
    x[2, 0, 0, 0] = -0.1
    x[3, 0, 0, 0] = float("nan")
    x[4, 0, 0, 0] = float("inf")
    out = m.ColorFramePosteriorNet()(x)
    assert out["valid"].tolist() == [False] * 5 + [True]
    assert (out["pred"] > 0).all()
    assert torch.isfinite(out["pred"]).all()
    assert torch.isfinite(out["transport_risk"]).all()
    assert (out["invalid_posterior_mass"][:5] == 1).all()
    bad = {key: val[:5] if isinstance(val, torch.Tensor) and val.ndim else val for key, val in out.items()}
    with pytest.raises(ValueError, match="valid"):
        m.posterior_nll(bad, torch.ones(5, 3))
    per_row = m.posterior_nll(out, torch.ones(6, 3), reduction="none")
    assert torch.isnan(per_row[:5]).all() and torch.isfinite(per_row[5])


def test_nonpositive_raw_point_is_refused_without_clipping():
    m = core()
    net = m.ColorFramePosteriorNet()
    with torch.no_grad():
        net.point_head.bias.fill_(-100)
    out = net(scene(torch.float32))
    assert (~out["valid"]).all()
    assert out["frame_valid"].all()
    assert (out["raw_pred"] < 0).all()
    assert (out["pred"] > 0).all()
    assert torch.isfinite(m.posterior_nll(out, torch.ones(2, 3)))


def test_vmf_log_normalizer_and_sphere_density_integral():
    m = core()
    k = torch.tensor([0., 1e-7, .1, 1., 10., 1000.], dtype=torch.float64, requires_grad=True)
    logc = m.vmf_log_normalizer(k)
    torch.testing.assert_close(logc[0], torch.tensor(-math.log(4 * math.pi), dtype=k.dtype))
    expected = torch.log(k[2:5] / (4 * math.pi * torch.sinh(k[2:5])))
    torch.testing.assert_close(logc[2:5], expected)
    z = torch.linspace(-1, 1, 20001, dtype=k.dtype)
    density = (logc[:5, None] + k[:5, None] * z).exp()
    integral = 2 * math.pi * torch.trapezoid(density, z, dim=-1)
    torch.testing.assert_close(integral, torch.ones(5, dtype=k.dtype), atol=1e-6, rtol=1e-6)
    logc.sum().backward()
    assert torch.isfinite(k.grad).all()


def test_vmf_large_float32_concentration_has_finite_gradient():
    m = core()
    concentration = torch.tensor([1e20], requires_grad=True)
    logc = m.vmf_log_normalizer(concentration)
    assert torch.isfinite(logc).all()
    logc.sum().backward()
    assert torch.isfinite(concentration.grad).all()


def test_canonical_target_solves_frame_and_nll_matches_single_component_density():
    m = core()
    frame = torch.tensor([[[1., .2, .1], [.1, 1.2, .2], [.2, .1, .9]]], dtype=torch.float64)
    q = F.normalize(torch.tensor([[.2, .5, .3]], dtype=frame.dtype), dim=-1)
    gt = (frame @ q.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(m.canonical_target(frame, gt), q)
    out = {"frame": frame, "frame_valid": torch.tensor([True]), "directions": q[:, None, :], "logits": torch.zeros(1, 1, dtype=q.dtype), "concentration": torch.tensor([[3.]], dtype=q.dtype)}
    expected = -math.log(3 / (4 * math.pi * math.sinh(3))) - 3
    torch.testing.assert_close(m.posterior_nll(out, gt), torch.tensor(expected, dtype=q.dtype))


def test_camera_density_applies_spherical_jacobian_and_is_scale_independent():
    m = core()
    frame = torch.diag_embed(torch.tensor([[2., 1., 1.]], dtype=torch.float64))
    q = torch.tensor([[1., 0., 0.]], dtype=frame.dtype)
    gt = torch.tensor([[1., 1e-12, 1e-12]], dtype=frame.dtype)
    out = {"frame": frame, "frame_valid": torch.tensor([True]), "directions": q[:, None], "logits": torch.zeros(1, 1, dtype=q.dtype), "concentration": torch.ones(1, 1, dtype=q.dtype)}
    canonical = m.posterior_nll(out, gt)
    camera = m.camera_posterior_nll(out, gt)
    torch.testing.assert_close(camera - canonical, torch.tensor(-math.log(4), dtype=q.dtype))
    scaled = dict(out, frame=frame * 7)
    torch.testing.assert_close(m.camera_posterior_nll(scaled, gt), camera)


def test_target_and_both_likelihoods_ignore_extreme_positive_scalar_magnitudes():
    m = core()
    frame = torch.tensor([[[1., .2, .1], [.1, 1.2, .2], [.2, .1, .9]]], dtype=torch.float64)
    gt = torch.tensor([[.4, .8, .5]], dtype=frame.dtype)
    q = m.canonical_target(frame, gt)
    out = {"frame": frame, "frame_valid": torch.tensor([True]), "directions": q[:, None], "logits": torch.zeros(1, 1, dtype=q.dtype), "concentration": torch.full((1, 1), 3., dtype=q.dtype)}
    expected_nll = m.posterior_nll(out, gt)
    expected_camera_nll = m.camera_posterior_nll(out, gt)
    for gt_scale, frame_scale in [(1e-15, 1.), (1e-100, 1.), (1e200, 1.), (1e-200, 1e200), (1e200, 1e-200)]:
        changed = dict(out, frame=frame * frame_scale)
        target = m.canonical_target(changed["frame"], gt * gt_scale)
        torch.testing.assert_close(target, q)
        torch.testing.assert_close(target.norm(dim=-1), torch.ones(1, dtype=q.dtype))
        torch.testing.assert_close(m.posterior_nll(changed, gt * gt_scale), expected_nll)
        torch.testing.assert_close(m.camera_posterior_nll(changed, gt * gt_scale), expected_camera_nll)


def test_complete_vmf_density_retains_large_concentration_peak_and_tiny_angle_penalty():
    m = core()
    concentration = torch.tensor([[1e20]], requires_grad=True)
    direction = torch.tensor([[[1., 0., 0.]]], requires_grad=True)
    out = {"frame": torch.eye(3, dtype=torch.float64)[None], "frame_valid": torch.tensor([True]), "directions": direction, "logits": torch.zeros(1, 1), "concentration": concentration}
    aligned_gt = torch.tensor([[1., 1e-30, 1e-30]], dtype=torch.float64)
    nll = m.posterior_nll(out, aligned_gt)
    expected_peak = math.log(2 * math.pi) - math.log(1e20)
    torch.testing.assert_close(nll, torch.tensor(expected_peak), atol=1e-5, rtol=1e-6)
    offset_gt = torch.tensor([[1., 1e-10, 1e-30]], dtype=torch.float64)
    offset_nll = m.posterior_nll(out, offset_gt)
    torch.testing.assert_close(offset_nll - nll, torch.tensor(.5), atol=1e-5, rtol=1e-5)
    offset_nll.backward()
    assert torch.isfinite(concentration.grad).all()
    assert torch.isfinite(direction.grad).all()


def test_near_tie_frame_switch_exposes_discontinuous_canonical_coordinates():
    m = core()
    x = scene()
    x[:, :, 0:4, 12:16] = x[:, :, 0:4, 0:4] * (1 - 1e-8)
    y = x.clone()
    y[:, :, 0:4, 12:16] = x[:, :, 0:4, 0:4] * (1 + 1e-8)
    net = m.ColorFramePosteriorNet().double()
    a, b = net(x), net(y)
    assert a["valid"].all() and b["valid"].all()
    assert a["frame_indices"].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert b["frame_indices"].tolist() == [[1, 2, 3], [1, 2, 3]]
    assert (a["frame_volume_margin"] < 1e-7).all()
    assert (b["frame_volume_margin"] < 1e-7).all()
    assert (a["canonical_mean"] - b["canonical_mean"]).abs().max() > .01


def test_random_positive_well_conditioned_mixing_preserves_canonical_features():
    m = core()
    torch.manual_seed(114)
    net = m.ColorFramePosteriorNet().double()
    x = scene()
    out = net(x)
    for _ in range(6):
        matrix = torch.eye(3, dtype=x.dtype) + torch.rand(3, 3, dtype=x.dtype) * .2
        mixed = net(torch.einsum("ij,njhw->nihw", matrix, x))
        assert mixed["valid"].all()
        torch.testing.assert_close(mixed["directions"], out["directions"], atol=2e-9, rtol=2e-9)


def test_risk_transport_depends_on_concentration_and_is_not_gl_invariant():
    m = core()
    frame = torch.eye(3, dtype=torch.float64)[None]
    pred = F.normalize(torch.ones(1, 3, dtype=frame.dtype), dim=-1)
    directions = F.normalize(torch.tensor([[[1., .6, 1.4]]], dtype=frame.dtype), dim=-1)
    logits = torch.zeros(1, 1, dtype=frame.dtype)
    k = torch.tensor([[1000.]], dtype=frame.dtype)
    risk, mass = m.transported_risk(frame, pred, directions, logits, k)
    matrix = torch.tensor([[[1., .8, .3], [.2, 1., .5], [.1, .2, 1.]]], dtype=frame.dtype)
    mixed_pred = F.normalize((matrix @ pred.unsqueeze(-1)).squeeze(-1), dim=-1)
    mixed_risk, _ = m.transported_risk(matrix, mixed_pred, directions, logits, k)
    assert abs(risk.item() - mixed_risk.item()) > 1
    diffuse_risk, diffuse_mass = m.transported_risk(frame, pred, directions, logits, k * .001)
    assert diffuse_mass.item() > mass.item()
    assert diffuse_risk.item() > risk.item()
    gains = torch.diag_embed(torch.tensor([[1.2, .7, 1.1]], dtype=frame.dtype))
    gain_pred = F.normalize((gains @ pred.unsqueeze(-1)).squeeze(-1), dim=-1)
    diagonal_risk, diagonal_mass = m.transported_risk(gains, gain_pred, directions, logits, k)
    torch.testing.assert_close(diagonal_risk, risk)
    torch.testing.assert_close(diagonal_mass, mass)
