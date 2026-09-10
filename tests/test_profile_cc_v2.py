"""Portable inference engineering checks, with synthetic pixels and fitted heads."""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def api():
    path = Path(__file__).resolve().parents[1] / "scripts/profile_cc_v2.py"
    spec = importlib.util.spec_from_file_location("profile_v2_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Toy(torch.nn.Module):
    def forward(self, x):
        anchor = x.mean((-2, -1)).clamp_min(1e-8)
        return torch.nn.functional.normalize(anchor, dim=-1), anchor[:, :1].repeat(1, 64)


def fixture():
    rng = np.random.default_rng(71)
    head = make_pipeline(StandardScaler(), Ridge(alpha=100)).fit(
        rng.normal(size=(100, 85)), rng.uniform(0.2, 1.5, 100)
    )
    return {"model": head, "scale": 1.7, "block": "combined"}


def test_embedded_ridge_matches_sklearn_and_rejects_invalid():
    module = api()
    torch.set_num_threads(4)
    payload = fixture()
    wrapper = module.PortableCC(Toy(), payload, threshold=100, head_dtype=torch.float64).eval()
    x = torch.rand(4, 3, 128, 128)
    with torch.inference_mode():
        pred, score, valid, accept = wrapper(x)
        p, context = wrapper.model(x)
        feature = module.risk_features_invariant(x, p, context)["combined"].numpy()
    expected = (
        module.selector.raw_predict(payload["model"], feature.astype(np.float64)) * payload["scale"]
    )
    np.testing.assert_allclose(score.numpy(), expected, rtol=1e-12, atol=1e-12)
    assert valid.all() and accept.all() and torch.isfinite(pred).all()
    invalid = x.clone()
    invalid[0] = 0
    invalid[1, 1] = 0
    invalid[2, 0, 0, 0] = float("nan")
    invalid[3, 0, 0, 0] = -1
    with torch.inference_mode():
        pred, score, valid, accept = wrapper(invalid)
    assert not valid.any() and not accept.any()
    assert torch.isfinite(pred).all() and torch.isfinite(score).all()


def test_portable_onnx_dynamic_batch_and_invalid_contract(tmp_path):
    module = api()
    wrapper = module.PortableCC(Toy(), fixture(), threshold=100).eval()
    rows = torch.rand(4, 3, 128, 128)
    report = module.export_and_check(wrapper, rows, tmp_path / "portable.onnx")
    assert report["passed"]
    assert report["floating_dtypes"] == ["FLOAT"]
    assert set(report["cases"]) == {
        "real_batch1",
        "real_batch4",
        "invalid_batch4",
        "infinite_batch1",
    }
