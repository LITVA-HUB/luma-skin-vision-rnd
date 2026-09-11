import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skin_spatial_model import grid_adjacency, screened_diffusion, SpatialColor, ARMS


def test_grid_is_symmetric_with_no_wraparound():
    a=grid_adjacency()
    assert torch.equal(a,a.T) and a.diag().sum()==0
    assert a[0].sum()==2 and a[9].sum()==4 and a[7,8]==0
    assert a.sum()==224


def test_diffusion_matches_direct_screened_system_and_preserves_bounds():
    torch.manual_seed(3)
    h=torch.randn(2,64,5,dtype=torch.float64)
    a=torch.rand(2,64,1,dtype=torch.float64)+.2
    w=grid_adjacency().double()[None].expand(2,-1,-1)*.3
    z=screened_diffusion(h,a,w,300)
    matrix=torch.diag_embed(a[...,0]+w.sum(-1))-w
    expected=torch.linalg.solve(matrix,a*h)
    torch.testing.assert_close(z,expected,atol=1e-10,rtol=0)
    assert (z>=h.amin(1,keepdim=True)-1e-12).all()
    assert (z<=h.amax(1,keepdim=True)+1e-12).all()
    constant=h[:,0:1].expand_as(h)
    torch.testing.assert_close(screened_diffusion(constant,a,w,3),constant)


def test_diffusion_gradients():
    torch.manual_seed(4)
    h=torch.randn(1,4,2,dtype=torch.float64,requires_grad=True)
    a=torch.full((1,4,1),.8,dtype=torch.float64,requires_grad=True)
    w=torch.full((1,4,4),.1,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x,y,z:screened_diffusion(x,y,z,3),(h,a,w))


def test_zero_steps_exactly_plain_and_plain_permutation_invariant():
    torch.manual_seed(5);m=SpatialColor('plain');x=torch.rand(2,64,18)
    reference=m(x)[0]
    torch.testing.assert_close(m(x[:,torch.randperm(64)])[0],reference,atol=1e-7,rtol=1e-6)
    m.arm='graph1';m.steps=0
    torch.testing.assert_close(m(x)[0],reference,atol=0,rtol=0)


def test_matched_initialization_and_spatial_effect():
    states=[];counts=[]
    for arm in ARMS:
        torch.manual_seed(6);model=SpatialColor(arm)
        states.append(model.state_dict());counts.append(sum(p.numel() for p in model.parameters()))
    assert len(set(counts))==1
    for state in states[1:]:
        for name,p in state.items():assert torch.equal(p,states[0][name])
    torch.manual_seed(6);graph=SpatialColor('graph3');scrambled=SpatialColor('graph3_scrambled')
    scrambled.load_state_dict(graph.state_dict());x=torch.rand(2,64,18)
    assert not torch.equal(graph(x)[0],scrambled(x)[0])
