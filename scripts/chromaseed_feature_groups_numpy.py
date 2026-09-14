"""NumPy-only subset consumer; original raw/X schemas use their frozen consumer."""

from __future__ import annotations

import numpy as np
from chromaseed_projection_numpy import Predictor as OriginalPredictor


class Predictor:
    def __init__(self, model):
        self.original = None
        if "feature_indices" not in model:
            self.original = OriginalPredictor(model)
            return
        indices = np.asarray(model["feature_indices"])
        if (
            indices.ndim != 1
            or indices.dtype != np.uint8
            or not 1 <= len(indices) < 36
            or np.any(indices >= 36)
            or np.any(np.diff(indices.astype(int)) <= 0)
        ):
            raise ValueError("increasing distinct uint8 indices inside color36 required")
        base = {
            "feature_indices",
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
            "centers",
            "coefficient",
            "width",
        }
        extra = {"correction", "gate_beta", "gate_mode", "rho"}
        if set(model) not in (base, base | extra):
            raise ValueError("unknown or incomplete subset schema")
        for k, v in model.items():
            dtype = np.uint8 if k in ("feature_indices", "gate_mode") else np.float32
            if np.asarray(v).dtype != dtype or not np.isfinite(v).all():
                raise ValueError("finite arrays with prescribed storage types required")
        if model["centers"].ndim != 2:
            raise ValueError("matrix centers required")
        n, d = model["centers"].shape
        shapes = dict(
            x_mean=(d,), x_std=(d,), y_mean=(3,), y_std=(3,), coefficient=(n, 3), width=()
        )
        if "correction" in model:
            shapes.update(correction=(n, 3), gate_beta=(d + 1,), gate_mode=(), rho=())
        if (
            n < 1
            or d != len(indices)
            or any(model[k].shape != shape for k, shape in shapes.items())
        ):
            raise ValueError("incompatible subset array shapes")
        if np.any(model["x_std"] <= 0) or np.any(model["y_std"] <= 0) or float(model["width"]) <= 0:
            raise ValueError("positive normalization and width required")
        self.indices = indices.copy()
        self.x_mean, self.x_std, self.y_mean, self.y_std = [
            model[k].copy() for k in ("x_mean", "x_std", "y_mean", "y_std")
        ]
        self.centers = model["centers"].astype(np.float64)
        self.norms = np.sum(self.centers * self.centers, axis=1)
        self.coefficient = model["coefficient"].astype(np.float64)
        self.denominator = d * float(model["width"]) ** 2
        self.correction = None
        if "correction" in model:
            if int(model["gate_mode"]) != 1 or not 0 <= float(model["rho"]) <= 1:
                raise ValueError("continuous gate and valid strength required")
            self.correction = model["correction"].astype(np.float64)
            self.beta = model["gate_beta"].astype(np.float64)
            self.rho = float(model["rho"])

    @property
    def cached_array_bytes(self):
        if self.original is not None:
            return self.original.cached_array_bytes
        return sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray))

    def __call__(self, color):
        if self.original is not None:
            return self.original(color)
        color = np.asarray(color, np.float32)
        if color.shape != (36,) or not np.isfinite(color).all():
            raise ValueError("one finite color36 vector required")
        z = ((color[self.indices] - self.x_mean) / self.x_std).astype(np.float64)
        distance = np.maximum(np.sum(z * z) + self.norms - 2 * (self.centers @ z), 0)
        kernel = np.exp(-0.5 * distance / self.denominator)
        value = kernel @ self.coefficient
        if self.correction is not None:
            score = min(max(float(z @ self.beta[1:] + self.beta[0]), -1), 1)
            value += self.rho * score * (kernel @ self.correction)
        return value * self.y_std + self.y_mean
