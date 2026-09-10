import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from cc_v7_core import SemanticColorNet, sensor_matrix, sensor_transform, teacher_render

from luma_skin_vision.cc.v2 import CompactResidualCC


def test_virtual_sensor_matches_independent_spectral_integration():
    gen=torch.Generator().manual_seed(81)
    sensitivity=torch.rand(3,31,generator=gen,dtype=torch.float64)
    illuminant=torch.rand(31,generator=gen,dtype=torch.float64)
    reflectance=torch.rand(7,31,generator=gen,dtype=torch.float64)
    rgb=(reflectance*illuminant)@sensitivity.T
    gt=sensitivity@illuminant
    m=sensor_matrix(1,generator=gen,dtype=torch.float64,identity_fraction=0)
    x=rgb.T.reshape(1,3,1,7)
    transformed,label=sensor_transform(x,gt[None],m)
    new_sensitivity=m[0]@sensitivity
    reference=(reflectance*illuminant)@new_sensitivity.T
    torch.testing.assert_close(transformed[0,:,0].T,reference)
    torch.testing.assert_close(label[0],F.normalize(new_sensitivity@illuminant,dim=0))
    assert (m>=0).all() and torch.linalg.det(m).abs().min()>0


def test_sensor_identity_and_seed_replay():
    a=sensor_matrix(8,generator=torch.Generator().manual_seed(2))
    b=sensor_matrix(8,generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    identity=sensor_matrix(2,generator=torch.Generator().manual_seed(2),identity_fraction=1)
    torch.testing.assert_close(identity,torch.eye(3)[None].expand(2,-1,-1))


def test_student_plain_forward_matches_existing_compact_direct():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    old=CompactResidualCC("direct","large").eval()
    model=SemanticColorNet().eval()
    model.load_state_dict(old.state_dict(),strict=False)
    x=torch.rand(2,3,64,64)
    with torch.no_grad():
        a=old(x)
        b=model(x)
    torch.testing.assert_close(a[0],b["pred"],rtol=0,atol=0)
    assert b["teacher_features"].shape==(2,16,384)
    assert sum(p.numel() for n,p in model.named_parameters() if not n.startswith("teacher_projection"))==sum(p.numel() for p in old.parameters())


def test_teacher_canonical_render_uses_gt_only_when_requested():
    x=torch.ones(1,3,32,32)*torch.tensor([.5,1.,2.])[None,:,None,None]
    gt=F.normalize(torch.tensor([[.5,1.,2.]]),dim=-1)
    raw=teacher_render(x,None)
    corrected=teacher_render(x,gt)
    assert raw.shape==corrected.shape==(1,3,224,224)
    assert torch.isfinite(corrected).all()
    # Undo normalizing constants: a flat scene becomes neutral after true GT correction.
    mean=torch.tensor([.485,.456,.406])[None,:,None,None]
    std=torch.tensor([.229,.224,.225])[None,:,None,None]
    rgb=corrected*std+mean
    torch.testing.assert_close(rgb[:,0],rgb[:,1])
    torch.testing.assert_close(rgb[:,1],rgb[:,2])


def test_teacher_role_guard_excludes_every_held_out_role():
    from cc_v7_teacher import training_indices
    rows=[{"id":str(i),"subset":role,"group":str(i)} for i,role in enumerate(["train","val","risk","cal","test"])]
    assert training_indices(rows).tolist()==[0]


def test_teacher_role_guard_rejects_group_leakage():
    import pytest
    from cc_v7_teacher import training_indices
    rows=[{"id":"a","subset":"train","group":"same"},{"id":"b","subset":"test","group":"same"}]
    with pytest.raises(ValueError,match="overlaps"):
        training_indices(rows)


def test_official_test_exception_is_explicit_and_cannot_include_fitting_roles():
    import pytest
    from cc_v7_teacher import training_indices
    rows=[{"id":"a","subset":"train","group":"same"},
          {"id":"b","subset":"test","official_split":"test","group":"same"}]
    assert training_indices(rows,allow_official_test_group_overlap=True).tolist()==[0]
    rows[1]["subset"]="val"
    with pytest.raises(ValueError,match="overlaps"):
        training_indices(rows,allow_official_test_group_overlap=True)
