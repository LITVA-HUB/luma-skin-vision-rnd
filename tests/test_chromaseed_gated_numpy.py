import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def payload():
    return dict(
        x_mean=np.zeros(36, np.float32),
        x_std=np.ones(36, np.float32),
        y_mean=np.zeros(3, np.float32),
        y_std=np.ones(3, np.float32),
        centers=np.zeros((1, 36), np.float32),
        coefficient=np.ones((1, 3), np.float32),
        width=np.array(1.0, np.float32),
        correction=np.full((1, 3), 0.5, np.float32),
        gate_beta=np.r_[0.0, 1.0, np.zeros(35)].astype(np.float32),
        gate_mode=np.array(2, np.uint8),
        rho=np.array(1.0, np.float32),
    )


def test_hard_gate_boundary_and_soft_saturation_have_known_answers():
    from chromaseed_gated_numpy import Predictor

    model = payload()
    hard = Predictor(model)
    for value in (-2.0, -1e-9, 0.0, 1e-9, 2.0):
        x = np.zeros(36, np.float32)
        x[0] = value
        expected = np.exp(-0.5 * float(x[0]) ** 2 / 36) * (1.0 + (0.5 if value >= 0 else -0.5))
        np.testing.assert_allclose(hard(x), np.full(3, expected), atol=1e-12, rtol=0)
    model["gate_mode"] = np.array(1, np.uint8)
    soft = Predictor(model)
    x = np.zeros(36, np.float32)
    x[0] = 2.0
    np.testing.assert_allclose(soft(x), hard(x), atol=1e-12, rtol=0)


def test_invalid_payload_and_input_fail_before_arithmetic():
    from chromaseed_gated_numpy import Predictor

    model = payload()
    model["gate_beta"] = np.zeros(36, np.float32)
    with pytest.raises(ValueError):
        Predictor(model)
    call = Predictor(payload())
    with pytest.raises(ValueError):
        call(np.zeros(35))
    with pytest.raises(ValueError):
        call(np.full(36, np.nan))


def test_standalone_module_does_not_import_training_frameworks():
    code = "import sys; sys.path.insert(0, 'scripts'); import chromaseed_gated_numpy; assert 'torch' not in sys.modules; assert 'scipy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)
