import importlib.util
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location('search_control', Path(__file__).parents[1]/'scripts/cc_v5_search_control.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Quadratic:
    search_offsets = torch.cartesian_prod(torch.linspace(-1, 1, 5), torch.linspace(-1, 1, 5))

    def __init__(self):
        self.queries = 0

    def query(self, cache, actions):
        self.queries += actions.shape[1]
        return {'angular_risk': ((actions-torch.tensor([.28, .28]))**2).sum(-1)}


def test_fixed_multiscale_is_exact_budget_and_never_recenters():
    model = Quadratic()
    cache = {'point_action': torch.zeros(1, 2), 'valid': torch.ones(1, dtype=torch.bool)}
    for steps, count in ((1, 25), (2, 51), (4, 103)):
        model.queries = 0
        result = module.fixed_select(model, cache, steps)
        assert model.queries == count == result['query_count']
        torch.testing.assert_close(result['action'], torch.tensor([[.24, .24]]))


def test_refused_input_does_not_take_arbitrary_corner():
    cache = {'point_action': torch.ones(1, 2), 'valid': torch.zeros(1, dtype=torch.bool)}
    result = module.fixed_select(Quadratic(), cache, 2)
    assert not result['valid'].any()
    torch.testing.assert_close(result['action'], torch.zeros(1, 2))
