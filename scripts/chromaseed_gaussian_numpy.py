"""Validated NumPy-only deterministic consumer for the TG mean network."""

from __future__ import annotations

import numpy as np


class Predictor:
    def __init__(self, model):
        required = {"w1", "b1", "w2", "b2", "x_mean", "x_std", "y_mean", "y_std"}
        if set(model) not in (required, required | {"feature_indices"}):
            raise ValueError("complete mean-network schema required")
        for key, value in model.items():
            dtype = np.uint8 if key == "feature_indices" else np.float32
            if np.asarray(value).dtype != dtype or not np.isfinite(value).all():
                raise ValueError("finite arrays with prescribed storage dtype required")
        if model["w1"].ndim != 2:
            raise ValueError("matrix first-layer weight required")
        hidden, d = model["w1"].shape
        shapes = dict(
            b1=(hidden,), w2=(3, hidden), b2=(3,), x_mean=(d,), x_std=(d,), y_mean=(3,), y_std=(3,)
        )
        if hidden < 1 or any(model[k].shape != v for k, v in shapes.items()):
            raise ValueError("incompatible network/normalizer shapes")
        self.indices = None
        if "feature_indices" in model:
            indices = model["feature_indices"]
            if (
                indices.shape != (d,)
                or not 1 <= d < 36
                or np.any(indices >= 36)
                or np.any(np.diff(indices.astype(int)) <= 0)
            ):
                raise ValueError("increasing distinct input coordinates required")
            self.indices = indices.copy()
        elif d != 36:
            raise ValueError("full input without indices must have dimension36")
        if np.any(model["x_std"] <= 0) or np.any(model["y_std"] <= 0):
            raise ValueError("positive normalization required")
        for key in ("x_mean", "x_std", "y_mean", "y_std"):
            setattr(self, key, model[key].copy())
        for key in ("w1", "b1", "w2", "b2"):
            setattr(self, key, model[key].astype(np.float64))

    @property
    def cached_array_bytes(self):
        return sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray))

    def __call__(self, color):
        color = np.asarray(color, np.float32)
        if color.shape != (36,) or not np.isfinite(color).all():
            raise ValueError("one finite color36 vector required")
        x = color if self.indices is None else color[self.indices]
        z = ((x - self.x_mean) / self.x_std).astype(np.float64)
        hidden = np.maximum(self.w1 @ z + self.b1, 0)
        return (self.w2 @ hidden + self.b2) * self.y_std + self.y_mean
