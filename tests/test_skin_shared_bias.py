import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from skin_shared_bias import paired_objective


def test_opposing_errors_cancel_only_in_mean_objective():
    p = torch.tensor([[1., 2, 3], [-1., -2, -3]], dtype=torch.float64, requires_grad=True)
    y = torch.zeros_like(p)
    ordinary = paired_objective(p, y, 'individual')
    assert ordinary.item() == pytest.approx(14/3)
    assert paired_objective(p, y, 'shared_only').item() == 0
    assert paired_objective(p, y, 'shared_half').item() == pytest.approx(7/3)
    assert paired_objective(p, y, 'consistency').item() == pytest.approx(28/3)
    paired_objective(p, y, 'shared_half').backward()
    torch.testing.assert_close(p.grad, p.detach()/6)


def test_common_bias_cost_identical_in_every_arm():
    p = torch.full((4, 3), 2., dtype=torch.float64)
    y = torch.zeros_like(p)
    assert all(paired_objective(p, y, arm).item() == 4 for arm in ['individual', 'shared_only', 'shared_half', 'consistency'])


def test_reference_and_pair_layout_are_checked():
    p = torch.zeros(4, 3); y = p.clone(); y[2, 0] = 1
    with pytest.raises(ValueError, match='reference'):
        paired_objective(p, y, 'shared_only')
    with pytest.raises(ValueError, match='even'):
        paired_objective(p[:3], p[:3], 'individual')
