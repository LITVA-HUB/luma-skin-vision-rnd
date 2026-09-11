import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from skin_local_teacher_features import pool_tokens, image_permutation
from skin_local_teacher_model import LocalTeacherColor
from skin_capture_model import CaptureColor


def test_pool_keeps_exact_two_by_two_spatial_correspondence():
    grid = (torch.arange(16)[:, None]*100+torch.arange(16)[None, :]).float()
    tokens = grid.reshape(1, 256, 1).expand(2, -1, 384)
    pooled = pool_tokens(tokens)
    expected = torch.tensor([[100*(2*r+.5)+(2*c+.5) for c in range(8)] for r in range(8)]).flatten()
    assert pooled.shape == (2, 64, 384)
    torch.testing.assert_close(pooled[0, :, 0], expected, atol=0, rtol=0)


def test_per_image_control_is_stable_and_preserves_every_token():
    a = image_permutation('example-a'); b = image_permutation('example-b')
    np.testing.assert_array_equal(np.sort(a), np.arange(64))
    np.testing.assert_array_equal(a, image_permutation('example-a'))
    assert not np.array_equal(a, b)


def test_no_teacher_matches_original_mixture_exactly():
    torch.manual_seed(17); base = CaptureColor('mixture').eval()
    torch.manual_seed(17); model = LocalTeacherColor('plain').eval()
    x = torch.rand(2, 64, 402)
    with torch.no_grad():
        for a, b in zip(base(x[..., :18].contiguous()), model(x), strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_alignment_modes_have_identical_parameter_shapes_and_raw_rgb():
    states = []
    for arm in ['plain', 'aligned', 'global', 'shuffled']:
        torch.manual_seed(17); model = LocalTeacherColor(arm); states.append(model.state_dict())
    assert all(all(torch.equal(states[0][k], v) for k, v in s.items()) for s in states)
    torch.manual_seed(5); x = torch.randn(2, 64, 402)
    model = LocalTeacherColor('global'); global_tokens = model.teacher_tokens(x)
    torch.testing.assert_close(global_tokens[:, 0], global_tokens[:, -1], atol=0, rtol=0)
    torch.testing.assert_close(global_tokens[:, 0], x[..., 18:].mean(1), atol=0, rtol=0)
