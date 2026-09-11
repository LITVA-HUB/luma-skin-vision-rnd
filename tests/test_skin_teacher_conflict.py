import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from skin_teacher_conflict import gradient_relation


def test_opposing_auxiliary_step_increases_primary_loss_to_first_order():
    x = torch.tensor(.5, requires_grad=True)
    color = torch.autograd.grad(x.square(), x, retain_graph=True)
    aux = torch.autograd.grad(.1*(x-2).square(), x)
    r = gradient_relation(color, aux)
    assert r['cosine'] == pytest.approx(-1)
    assert r['weighted_aux_to_color_norm_ratio'] == pytest.approx(.3)
    assert r['color_derivative_along_negative_aux_gradient'] == pytest.approx(.3)


def test_unused_auxiliary_parameters_are_zero_not_conflict():
    r = gradient_relation([torch.tensor([1., 2.])], [None])
    assert r['cosine'] is None
    assert r['weighted_aux_to_color_norm_ratio'] == 0
    assert r['color_derivative_along_negative_aux_gradient'] == 0
