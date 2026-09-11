import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skin_capture_model import delta_e00_squared,CaptureColor


def test_torch_ciede2000_matches_all_sharma_reference_pairs():
    fixture=np.loadtxt(Path(__file__).parent/'fixtures/ciede2000_sharma.txt')
    a=torch.tensor(fixture[:,:3],dtype=torch.float64);b=torch.tensor(fixture[:,3:6],dtype=torch.float64)
    np.testing.assert_allclose(delta_e00_squared(a,b).sqrt().numpy(),fixture[:,6],atol=5e-5,rtol=0)


def test_color_loss_gradients_match_finite_differences_away_from_hue_boundaries():
    a=torch.tensor([[55.,12.,18.],[34.,8.,9.]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[53.,14.,16.],[37.,9.,11.]],dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x:delta_e00_squared(x,b),a,eps=1e-5,atol=1e-4)


def test_zero_and_neutral_colors_have_finite_gradient_convention():
    a=torch.tensor([[50.,0.,0.],[50.,0.,0.],[50.,4.,8.]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[50.,0.,0.],[55.,3.,4.],[50.,4.,8.]],dtype=torch.float64)
    value=delta_e00_squared(a,b);assert value[0]==0 and value[2]==0
    value.sum().backward();assert torch.isfinite(a.grad).all()


def test_matched_capture_models_have_identical_parameters_and_uniform_gate_equivalence():
    torch.manual_seed(17);a=CaptureColor('uniform')
    torch.manual_seed(17);b=CaptureColor('mixture')
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    with torch.no_grad():
        a.gate.weight.zero_();a.gate.bias.zero_();b.gate.weight.zero_();b.gate.bias.zero_()
        x=torch.randn(2,64,18)
        torch.testing.assert_close(a(x)[0],b(x)[0],rtol=0,atol=0)
        p=a(x)[0];q=a(x[:,torch.randperm(64)])[0]
        torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
