"""Independent hat-trace and invariance tests for effective neural capacity."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def core():
    assert importlib.util.find_spec("chromaseed_neural_geometry"), "geometry implementation missing"
    from chromaseed_neural_geometry import degrees, spectrum

    return spectrum, degrees


def problem(coupled):
    rng = np.random.default_rng(733)
    z = np.column_stack((np.ones(19), rng.normal(size=(19, 5))))
    z[:, -1] = z[:, -2] + 2
    w = rng.uniform(0.2, 2, len(z))
    if not coupled:
        return z, w, None, np.sqrt(w)[:, None] * z, 1, 3
    q = rng.normal(size=(len(z), 3, 3))
    m = q @ q.transpose(0, 2, 1) + np.eye(3)[None]
    a = np.vstack(
        [np.kron(z[i : i + 1], np.sqrt(w[i]) * np.linalg.cholesky(m[i]).T) for i in range(len(z))]
    )
    return z, w, m, a, 3, 1


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("alpha", [0.1, 10, 1000])
def test_df_matches_full_hat_trace(coupled, alpha):
    spectrum, degrees = core()
    z, w, m, a, bias, multiplier = problem(coupled)
    g = a.T @ a
    penalty = np.diag(np.r_[np.zeros(bias), np.full(g.shape[0] - bias, alpha)])
    expected = multiplier * np.trace(np.linalg.solve(g + penalty, g))
    assert degrees(spectrum(z, w, m), alpha) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("coupled", [False, True])
def test_intercept_is_exempt_and_hidden_shift_changes_nothing(coupled):
    spectrum, degrees = core()
    z, w, m, _, _, _ = problem(coupled)
    original = spectrum(z, w, m)
    shifted = z.copy()
    shifted[:, 1:] += np.arange(5) * 7
    assert degrees(spectrum(shifted, w, m), 10) == pytest.approx(degrees(original, 10), abs=1e-9)
    assert degrees(original, 1e15) == pytest.approx(3, abs=1e-9)
    assert degrees(spectrum(np.ones((19, 1)), w, m), 10) == pytest.approx(3)


def test_invalid_metric_and_nonfinite_design_are_rejected():
    spectrum, _ = core()
    z, w, _, _, _, _ = problem(False)
    with pytest.raises(ValueError):
        spectrum(z, w, -np.repeat(np.eye(3)[None], len(z), axis=0))
    z[0, 1] = np.nan
    with pytest.raises(ValueError):
        spectrum(z, w)
