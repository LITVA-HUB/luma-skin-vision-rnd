"""Independent refit covers identity, compact and full-rotation paths."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_projection import fit_single, predict
from chromaseed_projection_reference import predict as reference_predict
from chromaseed_projection_reference import refit
from test_chromaseed_projection import toy


@pytest.mark.parametrize(
    "representation,family",
    [("raw", "norm_static"), ("d8_t01", "perceptual_joint_soft"), ("d36_t1", "norm_joint_soft")],
)
def test_svd_projection_dense_landmarks_qr_readout_matches_primary(representation, family):
    data = toy()
    model, _ = fit_single(*data, family, 17, representation, 0.1, rank=6)
    reference, info = refit(model, *data, family, 0.1, representation, 17, rank=6)
    assert info["actual_centers"] == 6
    np.testing.assert_allclose(
        reference_predict(reference, data[0]), predict(model, data[0]), rtol=0, atol=0.001
    )
