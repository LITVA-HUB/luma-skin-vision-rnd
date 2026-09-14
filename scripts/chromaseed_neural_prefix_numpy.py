"""NP exact ND prefix export and standalone NumPy one-color36 consumer."""

from __future__ import annotations

import numpy as np

SPECS = {
    "plain": (1, 36, 69),
    "local2": (2, 39, 32),
    "local4": (4, 39, 16),
    "blind4": (4, 39, 16),
    "e2e4": (4, 39, 16),
}
TOLERANCE = 0.10
PREP = (("x_mean", 36), ("x_std", 36), ("y_mean", 3), ("y_std", 3))


def capacity(payload):
    return dict(
        parameters=sum(v.size for k, v in payload.items() if k[0] in "wbvc"),
        numeric_bytes=sum(v.nbytes for v in payload.values() if v.dtype.kind in "biufc"),
        executed_blocks=sum(k.startswith("w") for k in payload),
    )


def choose(candidates):
    quality = min(
        candidates,
        key=lambda c: (c["clean"], c["p90"], c["numeric_bytes"], c["executed_blocks"], c["prefix"]),
    )
    allowed = [c for c in candidates if c["clean"] <= quality["clean"] + TOLERANCE]
    compact = min(
        allowed,
        key=lambda c: (c["numeric_bytes"], c["executed_blocks"], c["clean"], c["p90"], c["prefix"]),
    )
    return dict(quality=quality, compact=compact)


def export_prefix(parent, prefix):
    family = str(parent["family"])
    if family not in SPECS:
        raise ValueError("unknown family")
    k, d, h = SPECS[family]
    if (
        isinstance(prefix, (bool, np.bool_))
        or not isinstance(prefix, (int, np.integer))
        or not 1 <= prefix <= k
    ):
        raise ValueError("integer prefix in original schedule required")
    theta = np.asarray(parent["theta"])
    if theta.dtype != np.float32 or theta.shape != (k, (d + 1) * h + (h + 1) * 3):
        raise ValueError("invalid parent weights")
    if not np.isfinite(theta).all():
        raise ValueError("nonfinite parent weights")
    out = {key: np.asarray(parent[key]).copy() for key, _ in PREP}
    out.update(
        family=np.array(family), original_k=np.array(k, np.uint8), prefix=np.array(prefix, np.uint8)
    )
    indices = [prefix - 1] if family == "blind4" else range(prefix)
    for i, block in enumerate(indices):
        row = theta[block]
        out[f"w{i}"] = row[: d * h].reshape(d, h)[: 36 if i == 0 else d].copy()
        out[f"b{i}"] = row[d * h : (d + 1) * h].copy()
        out[f"v{i}"] = row[(d + 1) * h : -3].reshape(h, 3).copy()
        out[f"c{i}"] = row[-3:].copy()
    Predictor(out)
    return out


class Predictor:
    """Cached FP64 math, FP32 normalization, one finite (36,) input -> Lab3."""

    def __init__(self, payload):
        family = str(payload["family"])
        if family not in SPECS or np.asarray(payload["family"]).shape != ():
            raise ValueError("invalid family")
        k, _, h = SPECS[family]
        for key in ("original_k", "prefix"):
            v = np.asarray(payload[key])
            if v.shape != () or v.dtype != np.uint8:
                raise ValueError("uint8 scalar schedule metadata required")
        j = int(payload["prefix"])
        if int(payload["original_k"]) != k or not 1 <= j <= k:
            raise ValueError("invalid original schedule or prefix")
        count = 1 if family == "blind4" else j
        expected = {"family", "original_k", "prefix", *(key for key, _ in PREP)}
        expected.update(f"{key}{i}" for i in range(count) for key in "wbvc")
        if set(payload) != expected:
            raise ValueError("unexpected or missing payload arrays")
        self.prep = {}
        for key, size in PREP:
            v = np.asarray(payload[key])
            if (
                v.shape != (size,)
                or v.dtype != np.float32
                or not np.isfinite(v).all()
                or (key.endswith("std") and np.any(v <= 0))
            ):
                raise ValueError("invalid normalizer")
            self.prep[key] = v.copy()
        self.layers = []
        for i in range(count):
            layer = []
            for key, shape in zip(
                "wbvc", ((36 if i == 0 else 39, h), (h,), (h, 3), (3,)), strict=True
            ):
                v = np.asarray(payload[f"{key}{i}"])
                if v.shape != shape or v.dtype != np.float32 or not np.isfinite(v).all():
                    raise ValueError("invalid weights")
                layer.append(v.astype(np.float64))
            self.layers.append(tuple(layer))
        angle = np.arange(count) * np.pi / (2 * k)
        self.ratios = np.cos(angle[1:]) / np.cos(angle[:-1])
        self.scales = np.sin(angle[1:]) - self.ratios * np.sin(angle[:-1])
        self.cached_array_bytes = (
            sum(v.nbytes for v in self.prep.values())
            + sum(v.nbytes for layer in self.layers for v in layer)
            + self.ratios.nbytes
            + self.scales.nbytes
        )

    def _run(self, x):
        xx = ((x - self.prep["x_mean"]) / self.prep["x_std"]).astype(np.float64)
        state = None
        for i, (w, b, v, c) in enumerate(self.layers):
            incoming = xx if i == 0 else np.concatenate((xx, state), axis=-1)
            clean = np.maximum(incoming @ w + b, 0) @ v + c
            if i < len(self.layers) - 1:
                state = (
                    self.scales[i] * clean
                    if i == 0
                    else self.ratios[i] * state + self.scales[i] * clean
                )
        output = clean * self.prep["y_std"] + self.prep["y_mean"]
        if not np.isfinite(output).all():
            raise ValueError("nonfinite prediction")
        return output

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        if x.shape != (36,) or not np.isfinite(x).all():
            raise ValueError("one finite (36,) vector required")
        return self._run(x)


def predict(payload, x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty N x 36 required")
    return Predictor(payload)._run(x)
