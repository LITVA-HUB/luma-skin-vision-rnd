import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skin_mskcc_pixels import features
from skin_mskcc_vote import PatchVotes, aggregate
from skin_mskcc_vote_v2 import VoteAblation


def test_constant_color_preserved_and_no_nan_features():
    rgb=np.broadcast_to(np.array([64,128,192],dtype=np.uint8),(128,128,3)).copy()
    f=features(rgb)
    np.testing.assert_allclose(f['median'],[64/255,128/255,192/255],atol=1e-7)
    assert f['color'].shape==(36,) and f['hist'].shape==(512,) and f['tokens'].shape==(64,18)
    assert np.count_nonzero(f['hist'])==1 and f['hist'].sum()==1
    assert all(np.isfinite(v).all() for v in f.values())
    np.testing.assert_array_equal(f['color'][-6:],0)
    np.testing.assert_array_equal(f['tokens'][:,-3:],0)


def test_robust_updates_suppress_known_outlier_with_finite_gradient():
    votes=torch.tensor([[[0.,0.,0.]]*7+[[50.,0.,0.]]],requires_grad=True)
    logits=torch.zeros(1,8);std=torch.ones(3)
    plain,_=aggregate(votes,logits,std,0);robust,_=aggregate(votes,logits,std,3)
    assert plain[0,0]>6 and robust[0,0]<1
    robust.sum().backward();assert torch.isfinite(votes.grad).all()


def test_matched_capacity_initialization_and_permutation_invariance():
    torch.manual_seed(17);a=PatchVotes([10,5,5],0)
    torch.manual_seed(17);b=PatchVotes([10,5,5],3)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=torch.rand(2,64,18);perm=torch.randperm(64)
    for model in [a,b]:
        torch.testing.assert_close(model(x),model(x[:,perm]),rtol=1e-5,atol=1e-6)


def test_local_color_votes_cannot_take_global_answer_for_uniform_patches():
    torch.manual_seed(1);model=VoteAblation([10,5,5],local_only=True,steps=3)
    x=torch.rand(2,1,18).expand(-1,64,-1)
    before=model(x)
    with torch.no_grad():model.context[-2].bias.add_(10)
    torch.testing.assert_close(model(x),before,rtol=1e-5,atol=1e-6)


def test_constant_patch_training_has_finite_gradients():
    for local in [False,True]:
        model=VoteAblation([10,5,5],local_only=local,steps=3)
        loss=model(torch.ones(2,64,18)).square().mean();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
