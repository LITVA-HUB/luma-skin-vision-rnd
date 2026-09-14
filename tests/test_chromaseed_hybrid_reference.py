"""Independent dense/SVD/QR reconstruction of H's staged learning."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_hybrid import fit_bank, model_id, predict
from chromaseed_hybrid_reference import Basis, Geometry
from chromaseed_hybrid_reference import predict as reference_predict
from test_chromaseed_projection import toy


@pytest.mark.parametrize("two", [True, False])
def test_independent_staged_fit_and_predictions(two):
    data = toy(two)
    models, _ = fit_bank(*data, rank=4)
    b = Basis(Geometry(*data), 17, rank=4)
    for loss in ("norm", "perceptual"):
        for kind, alpha, rho, power in (
            ("raw", 0.1, 0.0, 0),
            ("projected", 1.0, 1.0, 0),
            ("blend", 0.1, 0.5, 0),
            ("uniform", 10.0, 0.25, 0),
            ("support", 1.0, 1.0, 1),
            ("support", 0.1, 0.5, 4),
        ):
            m = models[model_id(loss, 17, kind, alpha, rho, power)]
            independent = b.make(loss, kind, alpha, rho, power)
            np.testing.assert_allclose(
                reference_predict(m, data[0]), predict(m, data[0]), atol=2e-8, rtol=0
            )
            np.testing.assert_allclose(
                reference_predict(independent, data[0]), predict(m, data[0]), atol=0.001, rtol=0
            )


def test_reference_support_and_uniform_are_different_learned_paths():
    data = toy()
    b = Basis(Geometry(*data), 17, rank=4)
    a, c = b.make("norm", "uniform", 0.1, 0.5, 0), b.make("norm", "support", 0.1, 0.5, 4)
    assert np.max(np.abs(a["latent_coefficient"] - c["latent_coefficient"])) > 1e-4
