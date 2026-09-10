import numpy as np
import pytest
import torch

from luma_skin_vision.export import export_model
from luma_skin_vision.models import ColorRegressor


@pytest.mark.parametrize("method", ["baseline_c", "baseline_c_plus", "proposed_v1"])
def test_onnx_equivalence(tmp_path, method):
    pytest.importorskip("onnxruntime")
    model = ColorRegressor(method).eval()
    torch.manual_seed(1)
    # Avoid zero-output final layer hiding broken computation branches.
    torch.nn.init.normal_(model.head[-1].weight, std=0.01)
    x = torch.rand(2, 3, 64, 64)
    aux = torch.rand(2, 12)
    result = export_model(model, tmp_path / "model.onnx", x, aux)
    assert result["max_abs_lab_difference"] < 1e-3
    assert result["max_delta_e00_difference"] < 1e-3
    assert result["onnx_bytes"] > 0


def test_error_model_requires_heldout():
    from luma_skin_vision.uncertainty import fit_error

    with pytest.raises(ValueError, match="out-of-fold"):
        fit_error(np.ones((10, 2)), np.ones(10), provenance="fitted_training")
