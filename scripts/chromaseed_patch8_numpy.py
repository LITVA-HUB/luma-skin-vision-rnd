"""One-pass P8 consumer: NP color head plus a small local-patch branch."""

from __future__ import annotations

import numpy as np
from chromaseed_neural_prefix_numpy import Predictor as BasePredictor

EXTRA = {"u", "e", "g", "t_mean", "t_std"}


def transform_tokens(tokens, dose, anchor):
    x = np.asarray(tokens, np.float32)
    a = np.asarray(anchor, np.float64)
    if (
        x.shape[-2:] != (64, 18)
        or not np.isfinite(x).all()
        or a.shape != (3,)
        or not np.isfinite(a).all()
    ):
        raise ValueError("finite tokens64x18 and RGB anchor required")
    if not np.isfinite(dose) or not 0 <= dose <= 1 or np.any((a < 0) | (a > 1)):
        raise ValueError("bounded affine contraction required")
    if dose == 0:
        return x.copy()
    value = x.astype(np.float64)
    value[..., :12] += dose * (np.tile(a, 4) - value[..., :12])
    value[..., 12:] += dose * (-value[..., 12:])
    return value.astype(np.float32)


class Predictor:
    def __init__(self, model):
        self.patch = "u" in model
        base = {k: v for k, v in model.items() if k not in EXTRA}
        self.base = BasePredictor(base)
        if str(model["family"]) != "blind4" or model["w0"].shape != (36, 16):
            raise ValueError("NP643 head required")
        if set(model) != (set(base) | EXTRA if self.patch else set(base)):
            raise ValueError("incomplete local branch")
        self.cached_array_bytes = self.base.cached_array_bytes
        if self.patch:
            for key, shape in (
                ("u", (18, 8)),
                ("e", (8,)),
                ("g", (24, 16)),
                ("t_mean", (18,)),
                ("t_std", (18,)),
            ):
                v = np.asarray(model[key])
                if (
                    v.shape != shape
                    or v.dtype != np.float32
                    or not np.isfinite(v).all()
                    or (key == "t_std" and np.any(v <= 0))
                ):
                    raise ValueError("invalid local branch array")
                setattr(self, key, v.copy() if key.startswith("t_") else v.astype(np.float64))
                self.cached_array_bytes += getattr(self, key).nbytes

    def _run(self, x, tokens):
        if not self.patch:
            return self.base._run(x)
        t = (tokens - self.t_mean) / self.t_std
        encoded = np.maximum(t.astype(np.float64) @ self.u + self.e, 0)
        mean = encoded.mean(-2)
        spread = np.sqrt(((encoded - mean[..., None, :]) ** 2).mean(-2) + 1e-6)
        pooled = np.concatenate((mean, spread, encoded.max(-2)), axis=-1)
        prep = self.base.prep
        xx = ((x - prep["x_mean"]) / prep["x_std"]).astype(np.float64)
        w, b, v, c = self.base.layers[0]
        output = (np.maximum((xx @ w + b) + pooled @ self.g, 0) @ v + c) * prep["y_std"] + prep[
            "y_mean"
        ]
        if not np.isfinite(output).all():
            raise ValueError("nonfinite P8 prediction")
        return output

    def __call__(self, x, tokens=None):
        if not self.patch:
            return self.base(x)
        x, tokens = np.asarray(x, np.float32), np.asarray(tokens, np.float32)
        if (
            x.shape != (36,)
            or tokens.shape != (64, 18)
            or not np.isfinite(x).all()
            or not np.isfinite(tokens).all()
        ):
            raise ValueError("one finite color36 and tokens64x18 required")
        return self._run(x, tokens)


def predict(model, x, tokens=None):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty colorNx36 required")
    consumer = Predictor(model)
    if consumer.patch:
        tokens = np.asarray(tokens, np.float32)
        if tokens.shape != (len(x), 64, 18) or not np.isfinite(tokens).all():
            raise ValueError("matching finite tokensNx64x18 required")
    return consumer._run(x, tokens)


def choose(candidates):
    return min(
        candidates,
        key=lambda c: (c["clean"], c["p90"], c["numeric_bytes"], c["step"], c["lr"] or 0),
    )
