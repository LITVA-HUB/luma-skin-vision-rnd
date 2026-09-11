import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from skin_support_curve_interventions import intervene
from skin_support_curve import pixel_patches


def test_shuffle_preserves_each_patch_color_multiset():
    x=torch.arange(128*128*3).reshape(1,128,128,3).float()
    original=pixel_patches(x).flatten(2);changed=pixel_patches(intervene(x,'shuffle')).flatten(2)
    assert torch.equal(original.sort(2).values,changed.sort(2).values)
    assert not torch.equal(original,changed)


def test_mean_changes_only_within_patch_distribution():
    x=torch.rand(2,128,128,3)
    original=pixel_patches(x);changed=pixel_patches(intervene(x,'mean'))
    torch.testing.assert_close(original.mean((2,3)),changed.mean((2,3)),rtol=1e-6,atol=1e-7)
    assert changed.std((2,3)).max()==0
