import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
torch.set_num_threads(1)


def test_segmentation_shape_gradients_and_own_state_reload():
    from skin_face_segment import SkinUNet, skin_loss
    torch.manual_seed(123)
    model = SkinUNet(width=4)
    x = torch.rand(2,3,32,48)
    y = torch.randint(0,2,(2,1,32,48)).float()
    output = model(x)
    assert output.shape == y.shape
    loss = skin_loss(output,y)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    clone = SkinUNet(width=4)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(x),output,rtol=0,atol=0)
    with pytest.raises(ValueError, match='multiple'):
        model(torch.rand(1,3,33,32))


def test_skin_targets_retain_nose_and_exclude_other_classes():
    from skin_face_segment import binary_mask
    target = binary_mask(np.arange(11,dtype=np.uint8))
    assert target.tolist() == [0,1,0,0,0,0,1,0,0,0,0]
    with pytest.raises(ValueError, match='labels'):
        binary_mask(np.array([12]))


def test_overlap_scores_use_confusion_counts_and_defined_empty_case():
    from skin_face_segment import scores
    r = scores(tp=8,fp=2,fn=4,tn=10)
    assert r['iou'] == pytest.approx(8/14)
    assert r['dice'] == pytest.approx(16/22)
    assert r['precision'] == .8
    assert scores(0,0,0,10)['iou'] == 1


def test_deployment_parameter_budget():
    from skin_face_segment import SkinUNet
    count = sum(p.numel() for p in SkinUNet().parameters())
    assert 4_000_000 < count < 5_000_000
