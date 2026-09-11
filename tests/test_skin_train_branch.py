import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skin_train_branch_model import TrainingBranchColor, ARMS
from skin_spatial_model import SpatialColor


def test_all_strategies_deploy_exactly_plain_for_same_weights():
    x=torch.rand(2,64,18)
    torch.manual_seed(51);plain=SpatialColor('plain').eval()
    for arm in ARMS:
        torch.manual_seed(51);m=TrainingBranchColor(arm).eval()
        torch.testing.assert_close(m(x)[0],plain(x)[0],atol=0,rtol=0)
        assert m.active_parameters()==924932


def test_always_branch_matches_parent_training():
    x=torch.rand(2,64,18)
    for kind in ['graph','conv']:
        torch.manual_seed(52);parent=SpatialColor(kind+'3').train()
        torch.manual_seed(52);m=TrainingBranchColor(kind+'_always').train()
        torch.testing.assert_close(m(x)[0],parent(x)[0],atol=0,rtol=0)


def test_evaluation_does_not_consume_drop_rng_or_leave_steps_changed():
    torch.manual_seed(53);m=TrainingBranchColor('graph_drop').eval();x=torch.rand(2,64,18)
    before=torch.get_rng_state().clone();steps=m.steps;m(x)
    assert torch.equal(torch.get_rng_state(),before) and m.steps==steps
