"""P8 behavior: exact warm mapping, new local information and reproducible training."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_neural_prefix_numpy import predict as np_predict
from chromaseed_patch8_fit import Bank, fit, rate_factors, token_normalizers
from chromaseed_patch8_numpy import Predictor, choose, predict, transform_tokens


def fixture(n=12):
    rng = np.random.default_rng(713)
    x = rng.uniform(0.1, 0.9, (n, 36)).astype(np.float32)
    t = rng.uniform(0.1, 0.9, (n, 64, 18)).astype(np.float32)
    y = rng.normal([50, 10, 14], [4, 2, 3], (n, 3))
    prep = {}
    for key, v in (("x", x.astype(float)), ("y", y)):
        prep[key + "_mean"] = v.mean(0).astype(np.float32)
        prep[key + "_std"] = np.maximum(v.std(0), 1e-6).astype(np.float32)
    warm = []
    for s in (17, 29, 43):
        rr = np.random.default_rng(s)
        warm.append(
            dict(
                **{k: v.copy() for k, v in prep.items()},
                family=np.array("blind4"),
                original_k=np.array(4, np.uint8),
                prefix=np.array(2, np.uint8),
                w0=rr.uniform(-0.1, 0.1, (36, 16)).astype(np.float32),
                b0=np.zeros(16, np.float32),
                v0=rr.uniform(-0.1, 0.1, (16, 3)).astype(np.float32),
                c0=np.zeros(3, np.float32),
            )
        )
    return x, t, y, warm


def test_initial_mapping_sizes_and_fresh_export():
    x, t, y, warm = fixture()
    net = Bank(warm)
    prep = token_normalizers(t)
    assert net.theta.shape == (18, 1179)
    for slot in range(18):
        m = net.export(slot, warm, prep)
        np.testing.assert_array_equal(predict(m, x, t), np_predict(warm[slot // 6], x))
        patch = (slot % 6) // 3 == 1
        assert sum(v.nbytes for v in m.values() if v.dtype.kind in "biufc") == (
            5174 if patch else 2886
        )
        assert Predictor(m).cached_array_bytes == (9888 if patch else 5456)
    a = net.export(3, warm, prep)
    a["w0"].fill(100)
    assert not np.any(net.export(3, warm, prep)["w0"] == 100)


def test_local_inputs_affect_activated_branch_and_permutation_invariance():
    x, t, _, warm = fixture()
    m = Bank(warm).export(3, warm, token_normalizers(t))
    m["g"].fill(0.03)
    assert np.max(abs(predict(m, x, t) - predict(m, x, np.flip(t, axis=1).copy()))) < 2e-8
    assert np.max(abs(predict(m, x, t) - predict(m, x, np.zeros_like(t)))) > 0.01


def independent_tokens(rgb):
    p = rgb.reshape(8, 16, 8, 16, 3).transpose(0, 2, 1, 3, 4).reshape(64, 16, 16, 3)
    flat = p.reshape(64, 256, 3)
    q = np.quantile(flat, [0.1, 0.5, 0.9], axis=1).transpose(1, 0, 2).reshape(64, 9)
    grad = (np.abs(np.diff(p, axis=1)).mean((1, 2)) + np.abs(np.diff(p, axis=2)).mean((1, 2))) / 2
    return np.concatenate((q, flat.mean(1), flat.std(1), grad), axis=1)


def test_joint_token_affine_law_on_generated_continuous_rgb():
    rgb = np.random.default_rng(2).uniform(size=(128, 128, 3))
    tokens = independent_tokens(rgb).astype(np.float32)
    for dose in (0.0, 4 / 255, 64 / 255):
        a = np.array([0.1, 0.6, 0.9])
        expected = independent_tokens(rgb * (1 - dose) + dose * a).astype(np.float32)
        np.testing.assert_allclose(transform_tokens(tokens, dose, a), expected, rtol=0, atol=6e-8)
    np.testing.assert_array_equal(transform_tokens(tokens, 0, np.ones(3)), tokens)


def test_token_normalizers_use_given_fit_rows_only():
    _, t, _, _ = fixture()
    p = token_normalizers(t[:4])
    np.testing.assert_array_equal(p["t_mean"], t[:4].astype(float).mean((0, 1)).astype(np.float32))
    assert not np.array_equal(p["t_mean"], token_normalizers(t)["t_mean"])
    assert np.all(token_normalizers(np.zeros_like(t))["t_std"] == np.float32(1e-6))


def test_constant_token_backward_is_finite_and_stats_branch_has_zero_gradient():
    x, t, _, warm = fixture()
    net = Bank(warm)
    with torch.no_grad():
        net.theta[:, 795:] = 0.01
    out = net(torch.tensor(np.repeat(x[:2][None], 18, axis=0)), torch.zeros(18, 2, 64, 18))
    out.square().sum().backward()
    assert torch.isfinite(net.theta.grad).all()
    for slot in range(18):
        if slot % 6 < 3:
            assert torch.count_nonzero(net.theta.grad[slot, 643:]) == 0


def test_fit_wakes_patch_and_keeps_stats_g_zero_cpu():
    x, t, y, warm = fixture()
    result, info = fit(x, t, y, np.ones(len(x)), warm, 4, (0, 4))
    assert info["trajectory_count"] == 18
    assert np.any(result[4][3]["g"] != 0)
    assert "g" not in result[4][0]
    assert np.isfinite(predict(result[4][3], x, t)).all()


def test_original_horizon_rate_prefix_and_cpu_no_cumulative_decay():
    np.testing.assert_array_equal(rate_factors(8), rate_factors(32768)[:8])
    assert rate_factors(32768)[-1] == np.float32(0.1)
    x, t, y, warm = fixture(6)
    _, info = fit(x, t, y, np.ones(len(x)), warm, 16, (0, 16))
    expected = (
        np.array([r for _ in range(6) for r in (0.0001, 0.0003, 0.001)], np.float32)
        * rate_factors(16)[-1]
    )
    np.testing.assert_array_equal(np.array(info["final_learning_rates"], np.float32), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fixed_shape_graph_replay_and_eager_parity():
    x, t, y, warm = fixture(6)
    a, _ = fit(x, t, y, np.ones(len(x)), warm, 8, (0, 8), "cuda", "cuda_graph")
    b, _ = fit(x, t, y, np.ones(len(x)), warm, 16, (0, 8, 16), "cuda", "cuda_graph")
    c, _ = fit(x, t, y, np.ones(len(x)), warm, 8, (0, 8), "cuda", "eager")
    for i in range(18):
        for k in a[8][i]:
            np.testing.assert_array_equal(a[8][i][k], b[8][i][k])
            if a[8][i][k].dtype.kind == "f":
                np.testing.assert_allclose(a[8][i][k], c[8][i][k], rtol=0, atol=2e-6)
        np.testing.assert_allclose(
            predict(a[8][i], x, t), predict(c[8][i], x, t), rtol=0, atol=0.002
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward_matches_actual_numpy_local_consumer():
    x, t, _, warm = fixture()
    net = Bank(warm).cuda()
    with torch.no_grad():
        net.theta[:, 795:] = 0.02
        net.theta[:3, 795:] = 0
        net.theta[6:9, 795:] = 0
        net.theta[12:15, 795:] = 0
    prep = token_normalizers(t)
    xn = (x - warm[0]["x_mean"]) / warm[0]["x_std"]
    tn = (t - prep["t_mean"]) / prep["t_std"]
    with torch.no_grad():
        actual = (
            net(
                torch.tensor(np.repeat(xn[None], 18, axis=0), device="cuda"),
                torch.tensor(np.repeat(tn[None], 18, axis=0), device="cuda"),
            )
            .cpu()
            .numpy()
        )
    actual = actual * warm[0]["y_std"] + warm[0]["y_mean"]
    for slot in range(18):
        expected = predict(net.export(slot, warm, prep), x, t)
        np.testing.assert_allclose(actual[slot], expected, rtol=0, atol=0.002)


def test_selector_keeps_smaller_baseline_on_tie_but_prefers_quality():
    baseline = dict(clean=5.0, p90=9.0, numeric_bytes=2886, step=0, lr=None, arm="stats")
    patch = dict(clean=5.0, p90=9.0, numeric_bytes=5174, step=512, lr=0.0001, arm="patch8")
    assert choose([patch, baseline]) is baseline
    patch["clean"] = 4.9
    assert choose([patch, baseline]) is patch


def test_consumer_rejects_missing_branch_or_bad_normalizers():
    _, t, _, warm = fixture()
    m = Bank(warm).export(3, warm, token_normalizers(t))
    bad = {k: v for k, v in m.items() if k != "g"}
    with pytest.raises(ValueError):
        Predictor(bad)
    m["t_std"][0] = 0
    with pytest.raises(ValueError):
        Predictor(m)
