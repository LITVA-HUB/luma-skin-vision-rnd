"""The optimization study must preserve original trajectories and paired canaries."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_widen import fit as old_fit
from chromaseed_widen import rate_factors
from chromaseed_widen_early_fit import fit
from test_chromaseed_patch8 import fixture


def same(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_original_rate_cpu_preserves_weights_and_all_receipt_streams():
    x, t, y, warm = fixture()
    args = (x, t, y, np.ones(len(x)), warm, "m31")
    a, ia = old_fit(*args, 8, (0, 4, 8))
    b, ib = fit(*args, 0.0001, 8, (0, 4, 8))
    for step in a:
        for slot in range(6):
            same(a[step][slot], b[step][slot])
    assert ia["sampling_sha256"] == ib["sampling_sha256"]
    assert ia["slots"] == ib["slots"] and ib["schedule_horizon"] == 8192


def test_lower_rate_is_applied_without_affecting_high_rate_canaries():
    x, t, y, warm = fixture()
    args = (x, t, y, np.ones(len(x)), warm, "m31")
    a, _ = fit(*args, 0.0001, 8, (0, 4, 8))
    b, info = fit(*args, 0.00001, 8, (0, 4, 8))
    for step in a:
        for slot in (1, 3, 5):
            same(a[step][slot], b[step][slot])
    assert np.any(a[8][0]["w0"] != b[8][0]["w0"])
    rates = np.tile(np.array([0.00001, 0.001], np.float32), 3) * rate_factors(8)[-1]
    np.testing.assert_array_equal(np.array(info["final_learning_rates"], np.float32), rates)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA absent")
def test_largest_cuda_original_prefix_and_paired_canary_identity():
    x, t, y, warm = fixture(6)
    args = (x, t, y, np.ones(len(x)), warm, "m832")
    a, _ = old_fit(*args, 16, (0, 8, 16), "cuda", "cuda_graph")
    b, _ = fit(*args, 0.0001, 16, (0, 8, 16), "cuda", "cuda_graph")
    c, _ = fit(*args, 0.00003, 16, (0, 8, 16), "cuda", "cuda_graph")
    d, _ = fit(*args, 0.00003, 8, (0, 8), "cuda", "cuda_graph")
    for step in a:
        for slot in range(6):
            same(a[step][slot], b[step][slot])
            if slot % 2:
                same(b[step][slot], c[step][slot])
    for slot in range(6):
        same(c[8][slot], d[8][slot])


def test_unknown_rate_cannot_silently_change_bank_coordinates():
    x, t, y, warm = fixture()
    with pytest.raises(ValueError):
        fit(x, t, y, np.ones(len(x)), warm, "m31", 0.0002, 4, (0, 4))
