"""LT synthetic variation and fixed-recipe continuation behavioral checks."""

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def mod():
    assert importlib.util.find_spec("chromaseed_long_training_fit"), "missing LT implementation"
    return importlib.import_module("chromaseed_long_training_fit")


def fixture():
    from chromaseed_local_denoise import Bank, preprocessor
    from chromaseed_neural_prefix_numpy import export_prefix

    rng = np.random.default_rng(450)
    pixels = rng.uniform(0.15, 0.8, (18, 80, 3))
    q = (
        np.quantile(pixels, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], axis=1)
        .transpose(1, 0, 2)
        .reshape(18, 27)
    )
    x = np.concatenate((q, pixels.mean(1), pixels.std(1), np.zeros((18, 3))), axis=1).astype(
        np.float32
    )
    y = np.column_stack((40 + 25 * x[:, 27], 10 + 5 * x[:, 28], 15 + 9 * x[:, 29]))
    prep = preprocessor(x, y)
    warm = [export_prefix(Bank("blind4", [s]).export(0, prep), 2) for s in (17, 29, 43)]
    return x, y, np.arange(18), warm


def test_pool_preserves_original_prefix_and_moments():
    m = mod()
    x, _, rows, _ = fixture()
    pool = m.make_pool(x, rows)
    assert pool.shape == (18, 257, 36)
    np.testing.assert_array_equal(pool[:, 0], x)
    from chromaseed_gate_stability import basic_legal

    assert basic_legal(pool.reshape(-1, 36)).all()
    repeat = m.make_pool(x[[3, 0]], rows[[3, 0]])
    np.testing.assert_array_equal(repeat, pool[[3, 0]])
    draws = np.random.default_rng(880003 + 1009 * 3).random((256, 4))
    t = draws[:, 3] * 4 / 255
    np.testing.assert_allclose(
        pool[3, 1:, 27:30],
        (1 - t[:, None]) * x[3, 27:30] + t[:, None] * draws[:, :3],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        pool[3, 1:, 30:33], (1 - t[:, None]) * x[3, 30:33], rtol=0, atol=1e-7
    )
    np.testing.assert_array_equal(pool[3, 1:17], repeat[0, 1:17])


def test_variant_choice_keeps_clean_half_and_uniform_variant_mass():
    m = mod()
    u = torch.tensor([[0.0, 0.249, 0.499, 0.5, 0.625, 0.75, 0.875, 0.999]])
    for count in (0, 16, 256):
        result = m.variant_indices(u, torch.tensor([[count]])).numpy()[0]
        if count == 0:
            assert not result.any()
        else:
            np.testing.assert_array_equal(result[:3], [0, 0, 0])
            assert result[3] == 1 and result[-1] <= count and len(set(result[3:])) == 5


def test_sampling_and_learning_rate_prefixes_preserve_original_horizon():
    m = mod()
    a = m.uniforms((17, 29, 43), 7)
    b = m.uniforms((17, 29, 43), 31)
    np.testing.assert_array_equal(a, b[:, :7])
    np.testing.assert_array_equal(m.lr_factors(7), m.lr_factors(31)[:7])
    assert m.lr_factors(131072)[0] == 1 and abs(m.lr_factors(131072)[-1] - 0.1) < 1e-7


def test_pack_export_preserves_warm_start_without_training():
    m = mod()
    x, _, _, warm = fixture()
    net = m.HeadBank(warm)
    from chromaseed_neural_prefix_numpy import predict

    assert net.theta.shape == (18, 643)
    for i, slot in enumerate(m.SLOTS):
        actual = net.export(i, warm)
        expected = warm[(17, 29, 43).index(slot["seed"])]
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(predict(actual, x), predict(expected, x))


def test_cpu_continuation_saves_exact_baseline_and_valid_changed_weights():
    m = mod()
    x, y, rows, warm = fixture()
    models, info = m.fit(x, y, np.ones(len(x)), rows, warm, 3, (0, 1, 3))
    assert info["trajectory_count"] == 18 and info["original_rows"] == len(x)
    for key in warm[0]:
        np.testing.assert_array_equal(models[0][0][key], warm[0][key])
    assert not np.array_equal(models[3][0]["w0"], warm[0]["w0"])


def test_cpu_rate_schedule_uses_fixed_base_not_cumulative_decay():
    m = mod()
    x, y, rows, warm = fixture()
    _, info = m.fit(x, y, np.ones(len(x)), rows, warm, 128, (0, 128))
    expected = np.array([s["lr"] for s in m.SLOTS], np.float32) * m.lr_factors(128)[-1]
    np.testing.assert_array_equal(np.array(info["final_learning_rates"], np.float32), expected)


def test_selection_can_retain_baseline_and_prefers_smaller_mode_only_on_ties():
    from chromaseed_long_training_run import choose

    baseline = dict(clean=5.0, p90=8.0, step=0, variants=0, lr=None)
    same = dict(baseline, variants=256)
    worse = dict(clean=5.1, p90=8.0, step=131072, variants=256, lr=0.0003)
    assert choose([worse, same, baseline], True) == baseline
    better = dict(worse, clean=4.9)
    assert choose([baseline, better], True) == better


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fixed_bank_graph_reset_and_short_replay_are_exact():
    m = mod()
    x, y, rows, warm = fixture()
    full, _ = m.fit(x, y, np.ones(len(x)), rows, warm, 12, (0, 4, 12), "cuda", "cuda_graph")
    short, _ = m.fit(x, y, np.ones(len(x)), rows, warm, 4, (0, 4), "cuda", "cuda_graph")
    eager, _ = m.fit(x, y, np.ones(len(x)), rows, warm, 4, (0, 4), "cuda", "eager")
    for i in range(18):
        for key in full[4][i]:
            np.testing.assert_array_equal(full[4][i][key], short[4][i][key])
            if full[4][i][key].dtype.kind == "f":
                np.testing.assert_allclose(full[4][i][key], eager[4][i][key], rtol=2e-6, atol=2e-6)
            else:
                np.testing.assert_array_equal(full[4][i][key], eager[4][i][key])
