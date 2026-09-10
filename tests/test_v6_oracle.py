import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from cc_v4_experiment import action_rgb, oracle_errors
from cc_v6_oracle_audit import corrected_oracle


def test_oracle_search_clamps_residual_not_absolute_coordinates():
    point=np.array([[.15,0.]])
    anchor=np.array([[1.95,0.]])
    trajectory=np.tile(point[:,None],(1,4,1))
    gt=action_rgb(point+anchor+np.array([[.12,0.]]))
    actual=corrected_oracle(point,trajectory,anchor,gt)
    old=oracle_errors(point+anchor,trajectory+anchor[:,None],gt)
    assert actual[0][0] < 1e-6
    # First-stage clamped points differ; later stages include the base point.
    # Test the complete candidate coordinates, not merely a lucky minimum.
    assert actual[2].max() > 2
    assert old[0][0] > .05
