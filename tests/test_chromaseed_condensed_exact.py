import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.parametrize("kind", ["random", "duplicates", "large_offset", "constant", "single"])
def test_condensed_lower_median_equals_frozen_blocked_definition(kind):
    from chromaseed_condensed_exact import condensed_width
    from chromaseed_kernel import median_width
    rng = np.random.default_rng(314)
    x = rng.normal(size=(139, 36))
    if kind == "duplicates":
        x = np.repeat(x[:17], 8, axis=0)
    elif kind == "large_offset":
        x = np.repeat(x[:17], 8, axis=0) + 1e6
    elif kind == "constant":
        x = np.ones_like(x)
    elif kind == "single":
        x = x[:1]
    actual, info = condensed_width(x)
    assert actual == pytest.approx(median_width(x), abs=1e-12, rel=1e-12)
    assert info["pair_count"] == len(x) * (len(x) - 1) // 2


def test_condensed_fit_preserves_unseen_query_predictions():
    from chromaseed_condensed_exact import fit_condensed
    from chromaseed_fast_kernel import fit_one
    from chromaseed_kernel import predict_kernel
    rng = np.random.default_rng(138)
    x, y = rng.normal(size=(171, 36)).astype(np.float32), rng.normal(size=(171, 3))
    w = rng.uniform(.2, 2, size=len(x))
    expected, _ = fit_one(x, y, w, "column_exact", 64, 17, 1, 1)
    actual, _ = fit_condensed(x, y, w, 64, 17, 1, 1)
    q = rng.normal(size=(21, 36)).astype(np.float32)
    np.testing.assert_array_equal(actual["centers"], expected["centers"])
    np.testing.assert_allclose(predict_kernel(actual, q), predict_kernel(expected, q), rtol=0, atol=2e-6)
