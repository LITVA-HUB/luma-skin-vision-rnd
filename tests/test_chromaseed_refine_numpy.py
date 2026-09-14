import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.parametrize("family", ["stats_mlp", "patch_mlp", "recur_soft", "recur_top16", "recur_dynamic"])
def test_numpy_deployment_agrees_with_torch_and_really_exits(family):
    from chromaseed_refine import BankNet, apply_exit_policy, fit_preprocessor, transform
    from chromaseed_refine_numpy import NumpyRefiner

    torch.set_num_threads(1)
    rng = np.random.default_rng(385)
    x = rng.normal(size=(32, 36)).astype(np.float32)
    patches = rng.normal(size=(32, 64, 18)).astype(np.float32)
    y = (rng.normal(size=(32, 3)) * [10, 4, 6] + [50, 4, 10]).astype(np.float32)
    prep = fit_preprocessor(x, patches, y, np.ones(32))
    net = BankNet(family, [17]).eval()
    with torch.no_grad():
        net.layers["head"].weight.copy_(torch.from_numpy(rng.normal(size=(1, 48, 3)).astype(np.float32)))
    payload = {**prep, **net.export_slot(0)}
    deployed = NumpyRefiner(payload)
    xn, tn, base = transform(prep, x, patches)
    with torch.no_grad():
        expected, counts, _ = net(torch.from_numpy(xn)[None], torch.from_numpy(tn)[None], torch.from_numpy(base)[None])
    expected = expected[0].numpy() * prep["y_std"] + prep["y_mean"]
    for i in range(8):
        actual, connections = deployed.predict_trace(x[i], patches[i], threshold=0.)
        np.testing.assert_allclose(actual, expected[i], atol=1e-5, rtol=1e-6)
        np.testing.assert_array_equal(connections, counts[0, i].numpy())
        dense, _ = deployed.predict_trace(x[i], patches[i], threshold=0., sparse=False)
        np.testing.assert_allclose(actual, dense, atol=1e-5, rtol=1e-6)
        for threshold in (.25, 100.):
            selected, steps = apply_exit_policy(expected[i:i + 1], threshold)
            prediction, executed, _ = deployed.predict(x[i], patches[i], threshold)
            assert executed == steps[0]
            np.testing.assert_allclose(prediction, selected[0], atol=1e-5, rtol=1e-6)
