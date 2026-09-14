import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_iterative_selection_includes_unchanged_control_without_new_model():
    from chromaseed_perceptual import key
    from chromaseed_perceptual_train import candidate_grid
    grid = list(candidate_grid("midpoint_irls"))
    assert len(grid) == 36
    zero = [item for item in grid if item[2] == 0]
    assert len(zero) == 9
    for wi, ai, steps in zero:
        assert key("midpoint_irls", 17, wi, ai, steps) == key("norm_mse", 17, wi, ai, 0)


def test_selection_prefers_fewer_corrections_on_equal_inner_quality():
    from chromaseed_perceptual_train import choose_candidate
    candidates = [dict(person_mean=4., steps=s, alpha=a, width_factor=.5) for s, a in [(16, .1), (1, 1.), (0, 10.)]]
    assert choose_candidate(candidates)["steps"] == 0
