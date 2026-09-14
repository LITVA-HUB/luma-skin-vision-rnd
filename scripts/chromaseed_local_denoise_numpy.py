"""Actual target-free ND consumer: one finite color36 vector -> native Lab3."""

from __future__ import annotations

import numpy as np


class Predictor:
    def __init__(self, payload):
        sizes = {
            "plain": (1, 36, 69),
            "local2": (2, 39, 32),
            "local4": (4, 39, 16),
            "blind4": (4, 39, 16),
            "e2e4": (4, 39, 16),
        }
        self.family = str(payload["family"])
        if self.family not in sizes:
            raise ValueError("unknown ND family")
        self.k, d, h = sizes[self.family]
        theta = np.asarray(payload["theta"], np.float64)
        if theta.shape != (self.k, (d + 1) * h + (h + 1) * 3) or not np.isfinite(theta).all():
            raise ValueError("invalid ND parameters")
        self.prep = {}
        for key, size in (("x_mean", 36), ("x_std", 36), ("y_mean", 3), ("y_std", 3)):
            value = np.asarray(payload[key], np.float32)
            if (
                value.shape != (size,)
                or not np.isfinite(value).all()
                or (key.endswith("std") and np.any(value <= 0))
            ):
                raise ValueError("invalid normalizer")
            self.prep[key] = value.copy()
        self.layers = []
        for row in theta:
            self.layers.append(
                (
                    row[: d * h].reshape(d, h).copy(),
                    row[d * h : (d + 1) * h].copy(),
                    row[(d + 1) * h : -3].reshape(h, 3).copy(),
                    row[-3:].copy(),
                )
            )
        angle = np.arange(self.k + 1) * np.pi / (2 * self.k)
        self.a, self.s = np.sin(angle), np.cos(angle)
        self.a[0], self.s[0], self.a[-1], self.s[-1] = 0.0, 1.0, 1.0, 0.0
        self.cached_array_bytes = (
            sum(v.nbytes for v in self.prep.values())
            + sum(v.nbytes for layer in self.layers for v in layer)
            + self.a.nbytes
            + self.s.nbytes
        )

    def _run(self, x, trace):
        x = ((x - self.prep["x_mean"]) / self.prep["x_std"]).astype(np.float64)
        state = np.zeros(x.shape[:-1] + (3,), np.float64)
        outputs = []
        for t, (w1, b1, w2, b2) in enumerate(self.layers):
            visible = np.zeros_like(state) if self.family == "blind4" else state
            incoming = x if self.family == "plain" else np.concatenate((x, visible), axis=-1)
            clean = np.maximum(incoming @ w1 + b1, 0) @ w2 + b2
            if trace:
                outputs.append(clean * self.prep["y_std"] + self.prep["y_mean"])
            state = self.a[t + 1] * clean + self.s[t + 1] / self.s[t] * (state - self.a[t] * clean)
        output = (
            np.stack(outputs, axis=-2)
            if trace
            else clean * self.prep["y_std"] + self.prep["y_mean"]
        )
        if not np.isfinite(output).all():
            raise ValueError("nonfinite ND prediction")
        return output

    def trace(self, x):
        x = np.asarray(x, np.float32)
        if x.shape != (36,) or not np.isfinite(x).all():
            raise ValueError("one finite (36,) vector required")
        return self._run(x, True)

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        if x.shape != (36,) or not np.isfinite(x).all():
            raise ValueError("one finite (36,) vector required")
        return self._run(x, False)


def predict(payload, x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty N x 36 required")
    return Predictor(payload)._run(x, True)
