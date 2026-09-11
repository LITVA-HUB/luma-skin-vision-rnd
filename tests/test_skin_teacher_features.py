import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from skin_teacher_features import preprocess


def test_constant_rgb_has_exact_declared_normalization():
    rgb = np.broadcast_to(np.array([0, 128, 255], dtype=np.uint8), (2, 128, 128, 3)).copy()
    x = preprocess(rgb)
    expected = (torch.tensor([0., 128/255, 1])-torch.tensor([.485, .456, .406]))/torch.tensor([.229, .224, .225])
    assert x.shape == (2, 3, 224, 224)
    torch.testing.assert_close(x, expected[None, :, None, None].expand_as(x), atol=1e-6, rtol=0)


def test_input_contract_rejects_float_or_wrong_shape():
    with pytest.raises(ValueError):
        preprocess(np.zeros((2, 128, 128, 3)))
    with pytest.raises(ValueError):
        preprocess(np.zeros((2, 128, 128), dtype=np.uint8))
