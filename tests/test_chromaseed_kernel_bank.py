import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_small_bank_has_registered_configs_finite_predictions_and_valid_bounds():
    from chromaseed_kernel import AdaptiveKernel, predict_kernel
    from chromaseed_kernel_bank import (
        evaluate_bank,
        fit_bank,
        flatten_bank,
        make_adaptive,
        model_id,
    )

    rng = np.random.default_rng(4073)
    x, y = rng.normal(size=(38, 36)).astype(np.float32), rng.normal(size=(38, 3)) * [10, 4, 6] + [50, 4, 10]
    models, diagnostics, receipt = fit_bank(x, y, np.ones(len(x)), exact_backend="cpu")
    assert len(models) == receipt["readout_configurations"] == 369
    query = rng.normal(size=(7, 36)).astype(np.float32)
    predictions, violation = evaluate_bank(models, diagnostics, receipt, query, np.arange(7))
    assert violation <= 1e-6
    for key in list(models)[::37]:
        np.testing.assert_allclose(predictions[f"pred__{key}"], predict_kernel(models[key], query), rtol=0, atol=1e-10)
    arrays = flatten_bank(models, diagnostics)
    adaptive = AdaptiveKernel(make_adaptive(arrays, 1, 0, 17, 0.))
    expected = predictions[f"pred__{model_id('project_rpchol', 128, 17, 1, 0)}"]
    for i in range(7):
        prediction, used, bound, met = adaptive.predict_one(query[i])
        np.testing.assert_allclose(prediction, expected[i], rtol=0, atol=1e-10)
        assert used <= 38 and not met and np.isfinite(bound)
