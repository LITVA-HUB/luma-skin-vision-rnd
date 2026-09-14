"""Retained training must preserve block identities, noise and local credit."""

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def mod():
    assert importlib.util.find_spec("chromaseed_neural_blocks_fit"), "missing NB implementation"
    return importlib.import_module("chromaseed_neural_blocks_fit")


def data():
    rng = np.random.default_rng(361)
    x = rng.normal(size=(40, 36)).astype(np.float32)
    y = np.column_stack((50 + x[:, 0] * 3, 15 + x[:, 1], 20 + x[:, 2] * 2))
    return x, y.astype(np.float64), np.linspace(0.4, 1.6, len(x))


@pytest.mark.parametrize(
    "family,j,indices", [("local2", 1, [0]), ("local4", 3, [0, 1, 2]), ("blind4", 3, [2])]
)
def test_initialization_uses_original_block_and_seed(family, j, indices):
    from chromaseed_local_denoise import Bank

    m = mod()
    actual = m.RetainedBank(family, [29, 17], j)
    full = Bank(family, [29, 17])
    np.testing.assert_array_equal(actual.block_indices, indices)
    np.testing.assert_array_equal(
        actual.theta.detach().numpy().reshape(2, len(indices), -1),
        full.theta.detach().numpy().reshape(2, full.k, -1)[:, indices],
    )


def test_noise_retains_original_k_coordinates_not_shortened_rng_shape():
    from chromaseed_local_denoise_fit import noise_sequences

    m = mod()
    wanted = noise_sequences([17, 29, 17], 7, 4)[:, :, [0, 1]]
    actual = m.retained_noise([17, 29, 17], 7, 4, [0, 1])
    np.testing.assert_array_equal(actual, wanted)
    assert not np.array_equal(actual, noise_sequences([17, 29, 17], 7, 2))


def test_first_local_block_keeps_trainable_noise_input_rows():
    m = mod()
    net = m.RetainedBank("local4", [17], 1)
    net.local(torch.ones(1, 64, 36), torch.ones(1, 1, 64, 3)).square().mean().backward()
    assert net.d == 39
    assert torch.count_nonzero(net.theta.grad[0, : 39 * 16].reshape(39, 16)[36:]) > 0


@pytest.mark.parametrize(
    "field,value", [("prefix", np.array(1.5)), ("block_indices", np.array([1], np.uint8))]
)
def test_raw_export_rejects_wrong_original_identity(field, value):
    from chromaseed_local_denoise import preprocessor

    m = mod()
    x, y, _ = data()
    raw = m.RetainedBank("local4", [17], 1).export(0, preprocessor(x, y))
    raw[field] = value
    with pytest.raises(ValueError):
        m.to_prefix(raw)


@pytest.mark.parametrize(
    "family,j",
    [
        ("e2e4", 1),
        ("e2e4", 4),
        ("plain", 1),
        ("local4", 0),
        ("local4", 5),
        ("local4", 1.5),
        ("local4", True),
    ],
)
def test_invalid_or_coupled_subset_is_rejected(family, j):
    with pytest.raises(ValueError):
        mod().RetainedBank(family, [17], j)


@pytest.mark.parametrize("family,j", [("local2", 1), ("local4", 2), ("blind4", 3)])
def test_cpu_subset_training_matches_full_and_np_export(family, j):
    from chromaseed_local_denoise_fit import fit as full_fit
    from chromaseed_neural_prefix_numpy import export_prefix, predict

    m = mod()
    x, y, w = data()
    full, _ = full_fit(x, y, w, family, [(17, 0.003), (29, 0.001)], 12, (4, 12))
    subset, _ = m.fit(x, y, w, family, [(17, 0.003), (29, 0.001)], j, 12, (4, 12))
    for step in (4, 12):
        for slot in range(2):
            raw = subset[step][slot]
            np.testing.assert_allclose(
                raw["theta"], full[step][slot]["theta"][raw["block_indices"]], rtol=2e-6, atol=2e-6
            )
            expected = export_prefix(full[step][slot], j)
            actual = m.to_prefix(raw)
            np.testing.assert_allclose(predict(actual, x), predict(expected, x), rtol=0, atol=0.002)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("family,j", [("local4", 1), ("local4", 3), ("blind4", 2)])
def test_cuda_graph_reset_and_full_subset_semantics(family, j):
    from chromaseed_local_denoise_fit import fit as full_fit

    m = mod()
    x, y, w = data()
    full, _ = full_fit(
        x, y, w, family, [(17, 0.003), (29, 0.001), (43, 0.003)], 64, (64,), "cuda", "cuda_graph"
    )
    graph, _ = m.fit(
        x, y, w, family, [(17, 0.003), (29, 0.001), (43, 0.003)], j, 64, (64,), "cuda", "cuda_graph"
    )
    eager, _ = m.fit(
        x, y, w, family, [(17, 0.003), (29, 0.001), (43, 0.003)], j, 64, (64,), "cuda", "eager"
    )
    for slot in range(3):
        raw = graph[64][slot]
        np.testing.assert_allclose(
            raw["theta"], full[64][slot]["theta"][raw["block_indices"]], rtol=2e-6, atol=2e-6
        )
        np.testing.assert_allclose(raw["theta"], eager[64][slot]["theta"], rtol=2e-6, atol=2e-6)
