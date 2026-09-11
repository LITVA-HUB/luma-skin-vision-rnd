import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from skin_loss_field import risk_embedding, affine_weights, LossFieldImage


def test_risk_embedding_preserves_full_candidate_loss():
    rng=np.random.default_rng(31);cost=rng.random((8,37))*20
    phi,center,scale=risk_embedding(cost)
    weights=rng.dirichlet(np.ones(8),size=5);truth=np.eye(8)[[0,2,4,6,7]]
    expected=np.mean(((weights-truth)@cost)**2,axis=1)/(scale**2)
    actual=np.sum(((weights-truth)@phi)**2,axis=1)
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(center,cost.mean(0))


def test_affine_weights_remove_positivity_but_preserve_unit_sum():
    logits=torch.tensor([[1.,-3.,2.],[10.,-8.,0.]])
    w=affine_weights(logits)
    torch.testing.assert_close(w.sum(1),torch.ones(2))
    assert (w<0).any()
    torch.testing.assert_close(affine_weights(logits+5),w)


def test_shared_backbone_has_finite_gradients_for_field_heads():
    torch.manual_seed(17);model=LossFieldImage(12)
    p,g,h,logits=model(torch.rand(2,64,18))
    assert p.shape==(2,3) and h.shape==(2,4,3) and logits.shape==(2,12)
    loss=logits.softmax(1).square().sum()+p.square().mean()+g.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert 900_000<sum(p.numel() for p in model.parameters())<1_100_000
