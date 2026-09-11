import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from skin_capture_model import CaptureColor
from skin_expert_anchor import make_model,objective


def test_plain_removes_gate_and_preserves_initial_uniform_function():
    torch.manual_seed(31);old=CaptureColor('plain')
    torch.manual_seed(31);new=make_model('plain')
    x=torch.rand(3,64,18)
    torch.testing.assert_close(old(x)[0],new(x)[0],rtol=2e-6,atol=1e-7)
    assert sum(p.numel() for p in new.parameters())==924932
    assert not any('gate' in k for k in new.state_dict())
    assert new(x)[2].shape==(3,1,3)
    new(x)[0].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())


def test_conditional_anchor_uses_only_present_modes_and_unit_color_weight():
    p=torch.tensor([[2.,2.,2.]])
    h=torch.tensor([[[1.,1.,1.],[3.,3.,3.],[10.,10.,10.],[20.,20.,20.]]],requires_grad=True)
    g=torch.zeros(1,4);y=torch.zeros(1,3);q=torch.tensor([[.25,.75,0.,0.]])
    value=objective('conditional',p,g,h,y,q)
    expected=.5*4+.5*(.25*1+.75*9)+.1*torch.log(torch.tensor(4.))
    torch.testing.assert_close(value,expected)
    value.backward();assert torch.equal(h.grad[:,2:],torch.zeros_like(h.grad[:,2:]))


def test_uniform_anchor_control_is_independent_of_mode_at_equal_logits():
    p=torch.randn(2,3);g=torch.zeros(2,4);h=torch.randn(2,4,3);y=torch.randn(2,3)
    a=objective('uniform_anchor',p,g,h,y,torch.eye(4)[:2])
    b=objective('uniform_anchor',p,g,h,y,torch.eye(4)[2:])
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(a,.5*(p-y).square().mean()+.5*(h-y[:,None]).square().mean()+.1*torch.log(torch.tensor(4.)))


def test_plain_loss_has_no_capture_label_dependency():
    p=torch.randn(2,3,requires_grad=True);y=torch.randn(2,3);g=torch.randn(2,4,requires_grad=True)
    loss=objective('plain',p,g,p[:,None],y,torch.eye(4)[:2]);loss.backward()
    torch.testing.assert_close(loss,(p-y).square().mean())
    assert g.grad is None
