"""Grouping, transfer signs and exact restoration on an analytic regression."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_partition_preserves_examples_sizes_and_pooled_objective():
    from skin_gradient_transfer import partitions,group_means
    person=np.array(['a']*2+['b']*3+['c']*4)
    values=np.arange(18,dtype=float).reshape(9,2)
    found=partitions(person)
    assert len(found)==4
    for name,group in found:
        np.testing.assert_array_equal(np.bincount(group),[2,3,4])
        np.testing.assert_allclose(np.bincount(group)/9@group_means(values,group),values.mean(0))
        if name!='person':assert not np.array_equal(group,found[0][1])
    for (_,a),(_,b) in zip(found,partitions(person)):np.testing.assert_array_equal(a,b)


def test_stationary_pooled_gradient_does_not_imply_harmful_learning():
    from skin_gradient_transfer import gradient_statistics
    gram=np.array([[1.,-1.],[-1.,1.]])
    got=gradient_statistics(gram,np.array([.5,.5]))
    assert got['negative_pair_fraction']==1
    assert got['pooled_norm']==0
    assert got['mean_cosine']==-1


def test_finite_step_matches_quadratic_and_restores_on_exception():
    from skin_gradient_transfer import transient_step
    model=torch.nn.Linear(2,1,bias=False).double()
    with torch.no_grad():model.weight.copy_(torch.tensor([[.3,-.2]]))
    before={k:v.clone() for k,v in model.state_dict().items()}
    x=torch.tensor([[1.,2.],[-1.,1.]],dtype=torch.float64)
    y=torch.tensor([[.1],[.7]],dtype=torch.float64)
    def losses():return ((model(x)-y)**2).flatten()/2
    base=losses().detach()
    g=torch.autograd.grad(losses()[0],tuple(model.parameters()))[0].flatten()
    other=torch.autograd.grad(losses()[1],tuple(model.parameters()))[0].flatten()
    eps=1e-3;direction=-g/g.norm()
    with transient_step(model,g,eps):
        actual=losses().detach()-base
        expected=torch.stack([g@direction,other@direction])*eps+.5*eps**2*(x@direction)**2
        torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-10)
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
    try:
        with transient_step(model,g,eps):raise RuntimeError('probe')
    except RuntimeError:pass
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
