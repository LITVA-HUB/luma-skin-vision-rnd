import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_bank_batches_replay_across_learning_rates_and_preserve_weighting():
    from chromaseed_refine_train import sampling_indices

    weights = np.array([1., 3., 6.])
    sampled = sampling_indices(weights, [17, 29, 17], 256)
    np.testing.assert_array_equal(sampled[0], sampled[2])
    assert not np.array_equal(sampled[0], sampled[1])
    np.testing.assert_allclose(np.bincount(sampled[0].ravel(), minlength=3) / sampled[0].size, [.1, .3, .6], atol=.015)


def test_small_training_improves_nonlinear_fit_and_replays_exactly():
    from chromaseed_refine_train import predict_bank, train_bank

    torch.set_num_threads(1)
    rng = np.random.default_rng(57)
    x = rng.normal(size=(72, 36)).astype(np.float32)
    patches = rng.normal(size=(72, 64, 18)).astype(np.float32)
    y = np.column_stack((50 + 5 * x[:, 0] ** 2, 4 + 3 * x[:, 1] ** 2, 10 + 4 * x[:, 2] ** 2)).astype(np.float32)
    models, receipt = train_bank(x, patches, y, np.ones(len(x)), "stats_mlp", [(17, .003), (29, .003)], 80, (1, 80), "cpu")
    early, _ = predict_bank(models[1], x, patches, "cpu")
    late, _ = predict_bank(models[80], x, patches, "cpu")
    assert np.mean((late[:, :, -1] - y) ** 2) < .7 * np.mean((early[:, :, -1] - y) ** 2)
    again, _ = train_bank(x, patches, y, np.ones(len(x)), "stats_mlp", [(17, .003), (29, .003)], 80, (1, 80), "cpu")
    np.testing.assert_array_equal(models[80]["theta"], again[80]["theta"])
    assert receipt["steps"] == 80 and receipt["batch_size"] == 64
    with pytest.raises(ValueError):
        train_bank(x, patches, y, np.ones(len(x)), "stats_mlp", [(17, .003)], 8, (9,), "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph equivalence requires a CUDA GPU")
@pytest.mark.parametrize("family", ["stats_mlp", "patch_mlp", "recur_soft", "recur_top16", "recur_dynamic"])
def test_cuda_graph_matches_eager_updates_and_checkpoint_budget(family):
    from chromaseed_refine_train import setup, train_bank

    setup("cuda")
    rng = np.random.default_rng(351)
    x = rng.normal(size=(48, 36)).astype(np.float32)
    patches = rng.normal(size=(48, 64, 18)).astype(np.float32)
    y = (rng.normal(size=(48, 3)) * [10, 3, 5] + [50, 4, 10]).astype(np.float32)
    arguments = (x, patches, y, np.ones(len(x)), family, [(17, .0003), (29, .001), (17, .003)], 64, (1, 16, 64), "cuda")
    eager, _ = train_bank(*arguments, engine="eager")
    graph, receipt = train_bank(*arguments, engine="cuda_graph")
    for checkpoint in (1, 16, 64):
        np.testing.assert_allclose(graph[checkpoint]["theta"], eager[checkpoint]["theta"], atol=2e-6, rtol=2e-5)
    assert receipt["steps"] == 64 and receipt["engine"] == "cuda_graph"
