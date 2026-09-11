import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from skin_graph_support_model import GraphSupportColor


def test_inference_is_identical_core_for_all_branch_flags():
    torch.manual_seed(17);model=GraphSupportColor().eval();x=torch.rand(2,64,18)
    a=model(x,branch=False)[0];b=model(x,branch=True)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    perm=torch.randperm(64)
    torch.testing.assert_close(a,model(x[:,perm],branch=True)[0],rtol=1e-5,atol=1e-7)


def test_training_branch_receives_gradient_only_when_requested():
    model=GraphSupportColor().train();x=torch.rand(2,64,18)
    model(x,branch=False)[0].square().sum().backward()
    assert model.relation.weight.grad is None
    model.zero_grad(set_to_none=True);model(x,branch=True)[0].square().sum().backward()
    assert model.relation.weight.grad is not None and model.relation.weight.grad.abs().sum()>0
    assert model.steps==3
