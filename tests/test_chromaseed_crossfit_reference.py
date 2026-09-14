"""Independent native teacher decoding and QR correction reconstruction."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_crossfit import fit_bank
from chromaseed_crossfit_reference import head, teacher
from chromaseed_hybrid_reference import Basis, Geometry, predict
from test_chromaseed_crossfit import SETTINGS
from test_chromaseed_projection import toy


@pytest.mark.parametrize("two", [True, False])
def test_independent_teacher_and_native_residual_heads(two):
    data = toy(two)
    models, teachers, tables, rec = fit_bank(*data, SETTINGS, rank=4)
    geometry = Geometry(*data)
    basis = Basis(geometry, 17, rank=4)
    p = data[2]
    native = []
    for j, person in enumerate(rec["people"]):
        keep = p != person
        ref, _ = teacher(
            teachers[f"exclude{j}_perceptual_s17"],
            *(a[keep] for a in data),
            "perceptual",
            17,
            rank=4,
        )
        native.append(predict(ref, data[0]))
    native = np.stack(native)
    np.testing.assert_allclose(native, tables["native__perceptual_s17"], atol=0.001, rtol=0)
    for arm, indices in (
        ("out_person", tables["own_teacher"]),
        ("in_matched", tables["matched_teacher"]),
    ):
        routed = native[indices, np.arange(len(p))]
        for kind in ("uniform", "support"):
            m = head(basis, routed, "perceptual", SETTINGS["perceptual"][kind])
            expected = models[f"{arm}_perceptual_{kind}_s17"]
            np.testing.assert_allclose(
                predict(m, data[0]), predict(expected, data[0]), atol=0.001, rtol=0
            )
