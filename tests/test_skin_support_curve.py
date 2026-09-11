import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from skin_support_curve import SkinRepresentation, patient_roles, pixel_patches


def test_nested_people_keep_fixed_holdout_and_camera_proportions():
    people=np.repeat(np.arange(24),2)
    device=np.where(people<8,'SLR','ipod')
    previous=set()
    held=None
    for n in (6,12,18):
        train,hold=patient_roles(people,device,n,17)
        selected=set(people[train]); h=set(people[hold])
        assert len(selected)==n and len(h)==6 and selected.isdisjoint(h)
        assert previous<=selected
        assert len(set(people[train & (device=='SLR')]))==n//3
        assert held is None or held==h
        held=h;previous=selected


def test_patch_layout_preserves_pixels_without_cross_patch_context():
    rgb=torch.arange(128*128*3).reshape(1,128,128,3).float()
    p=pixel_patches(rgb)
    assert p.shape==(64,3,16,16)
    torch.testing.assert_close(p[9],rgb[0,16:32,16:32].permute(2,0,1))
    assert torch.equal(p.permute(0,2,3,1).reshape(8,8,16,16,3).permute(0,2,1,3,4).reshape(1,128,128,3),rgb)


def test_zero_adapters_start_from_same_color_and_receive_gradient():
    torch.set_num_threads(2)
    x=torch.rand(2,64,18);rgb=torch.rand(2,128,128,3)
    outputs=[];counts=[]
    for arm in ('baseline','statistics','pixels'):
        torch.manual_seed(17);m=SkinRepresentation(arm)
        p,_,_=m(x,rgb);outputs.append(p.detach())
        counts.append(sum(p.numel() for p in m.parameters()))
        if arm!='baseline':
            p.square().mean().backward()
            assert m.adapter[-1].weight.grad.abs().sum()>0
    assert torch.equal(outputs[0],outputs[1]) and torch.equal(outputs[0],outputs[2])
    assert abs(counts[1]-counts[2])/counts[1]<.001
