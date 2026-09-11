import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skin_nuisance_model import NuisanceColor, ARMS, control_residual
from skin_spatial_model import SpatialColor,grid_adjacency


def test_every_control_has_exact_plain_inference():
    x=torch.rand(2,64,18)
    torch.manual_seed(71);plain=SpatialColor('plain').eval()
    for arm in ARMS:
        torch.manual_seed(71);m=NuisanceColor(arm).eval()
        torch.testing.assert_close(m(x)[0],plain(x)[0],atol=0,rtol=0)
        assert m.active_parameters()==924932


def test_learned_control_matches_original_training_operator():
    torch.manual_seed(72);m=NuisanceColor('learned').train()
    torch.manual_seed(72);original=SpatialColor('graph3').train();x=torch.rand(2,64,18)
    torch.testing.assert_close(m(x)[0],original(x)[0],atol=0,rtol=0)


def test_fixed_operators_preserve_constants_and_global_is_permutation_equivariant():
    h=torch.randn(2,64,9);constant=h[:,:1].expand_as(h);adj=grid_adjacency()
    for kind in ['fixed_grid','global']:
        torch.testing.assert_close(control_residual(constant,kind,adj),torch.zeros_like(h),atol=1e-6,rtol=0)
    perm=torch.randperm(64)
    torch.testing.assert_close(control_residual(h[:,perm],'global',adj),control_residual(h,'global',adj)[:,perm],atol=1e-6,rtol=0)
    assert torch.equal(control_residual(h,'bias',adj),torch.full_like(h,.05))
