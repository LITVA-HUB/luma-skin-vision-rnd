"""Lazy batch-one AS/HR recurrence, with explicitly bounded execution.

Consumes exported features and weights only; no Torch or image processing.
The default executes all four trained passes. An optional positive L2 threshold
compares successive native Lab predictions: it is NOT a calibrated error bound,
confidence score, or an adopted early-exit policy. No payload exit threshold is
inherited. Validate any prospective policy on INNER data before evaluating it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.special import expit

VARIANTS = ("soft_small", "dynamic_small", "soft5m", "dynamic5m")
MODES = ("unit", "wide", "linear")


@dataclass(frozen=True)
class PassOutput:
    index: int
    prediction: np.ndarray
    change_l2: float | None
    active_tokens: int


@dataclass(frozen=True)
class PredictionTrace:
    passes: tuple[PassOutput, ...]
    stop_reason: str

    @property
    def prediction(self):
        return self.passes[-1].prediction

    @property
    def passes_executed(self):
        return len(self.passes)


def _array(value, shape, name, *, positive=False):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all() or (positive and np.any(array <= 0))):
        raise ValueError(f"invalid {name}: expected finite {shape}" + (" positive" if positive else ""))
    copy = array.copy()
    copy.setflags(write=False)
    return copy


def _silu(value):
    return value * expit(value)


class StreamingPredictor:
    """One region per call; generators share immutable weights, not request state."""

    def __init__(self, model):
        self.variant = str(model["variant"])
        self.mode = str(model.get("head_mode", "unit"))
        if self.variant not in VARIANTS or self.mode not in MODES:
            raise ValueError("only recurrent AS/HR exports and unit/wide/linear heads are supported")
        a, e, h = (32, 24, 48) if self.variant.endswith("small") else (384, 256, 1120)
        layout = (
            ("token1", 18, a), ("token2", a, e), ("context", 36 + e, h),
            ("query", h + 3, e), ("key", e, e),
            ("update1", 2 * h + e + 3, h), ("update2", h, h), ("head", h, 3),
        )
        count = sum((incoming + (name != "key")) * outgoing
                    for name, incoming, outgoing in layout)
        theta = _array(model["theta"], (count,), "theta").astype(np.float64)
        theta.setflags(write=False)
        self.layers = {}
        offset = 0
        for name, incoming, outgoing in layout:
            size = incoming * outgoing
            weight = theta[offset:offset+size].reshape(incoming, outgoing)
            offset += size
            bias = np.zeros(outgoing) if name == "key" else theta[offset:offset+outgoing]
            offset += 0 if name == "key" else outgoing
            bias.setflags(write=False)
            self.layers[name] = weight, bias
        self.base = {}
        for name, shape in (
            ("x_mean", (36,)), ("x_std", (36,)), ("y_mean", (3,)), ("y_std", (3,)),
            ("w0", (36, 16)), ("b0", (16,)), ("v0", (16, 3)), ("c0", (3,)),
        ):
            self.base[name] = _array(model["base_" + name], shape, name,
                                     positive=name.endswith("_std"))
        self.t_mean = _array(model["t_mean"], (18,), "t_mean")
        self.t_std = _array(model["t_std"], (18,), "t_std", positive=True)

    def _layer(self, name, value):
        weight, bias = self.layers[name]
        return value @ weight + bias

    def _head(self, state):
        value = self._layer("head", state)
        if self.mode == "unit":
            return np.tanh(value)
        if self.mode == "wide":
            return 4 * np.tanh(value / 4)
        return value

    def iter_passes(self, color, patches):
        """Lazily yield passes 1..4. Closing this iterator skips all remaining work.

        Shape and arithmetic follow the frozen singleton consumer, including its
        FP32 input conversion before normalization and FP64 subsequent layers.
        Token encoding and all 64 attention scores still execute. Dynamic masking
        does not skip those operations; savings here come only from omitted passes.
        """
        x = _array(color, (36,), "color").astype(np.float32)[None]
        tokens = _array(patches, (64, 18), "patches").astype(np.float32)[None]
        b = self.base
        xx = ((x - b["x_mean"]) / b["x_std"]).astype(np.float64)
        tt = ((tokens - self.t_mean) / self.t_std).astype(np.float64)
        if not np.isfinite(xx).all() or not np.isfinite(tt).all():
            raise ValueError("nonfinite normalized features")
        pred = np.maximum(xx @ b["w0"].astype(float) + b["b0"], 0) @ b["v0"].astype(float) + b["c0"]
        t = _silu(self._layer("token2", _silu(self._layer("token1", tt))))
        context = _silu(self._layer("context", np.concatenate((xx, t.mean(1)), -1)))
        keys = self._layer("key", t)
        state = context
        previous_native = None
        for index in range(1, 5):
            q = self._layer("query", np.concatenate((state, pred), -1))
            scores = np.sum(keys * q[:, None], -1) / math.sqrt(t.shape[-1])
            mask = np.ones_like(scores)
            if self.variant.startswith("dynamic"):
                mask = (scores > 0).astype(float)
                top = np.argsort(scores, axis=-1, kind="stable")[:, -4:]
                np.put_along_axis(mask, top, 1.0, axis=-1)
            attention = np.exp(scores - scores.max(-1, keepdims=True)) * mask
            attention /= attention.sum(-1, keepdims=True)
            pooled = np.sum(t * attention[..., None], 1)
            combined = np.concatenate((state, context, pooled, pred), -1)
            update = np.tanh(self._layer("update2", _silu(self._layer("update1", combined))))
            state = 0.5 * state + 0.5 * update
            pred = pred + 0.25 * self._head(state)
            native = (pred * b["y_std"].astype(float) + b["y_mean"].astype(float))[0]
            if not np.isfinite(native).all() or not np.isfinite(attention).all():
                raise ValueError("nonfinite recurrent prediction")
            change = None if previous_native is None else float(np.linalg.norm(native - previous_native))
            previous_native = native.copy()
            snapshot = native.copy()
            snapshot.setflags(write=False)
            yield PassOutput(index, snapshot, change, int(mask.sum()))

    def predict_trace(self, color, patches, *, max_passes=4, exit_threshold=0.0, min_passes=2):
        """Run a fixed prefix or an explicit, uncalibrated successive-change rule.

        Zero disables threshold stopping; positive thresholds can stop after at
        least min_passes, and only before the fixed budget is exhausted. This is
        Euclidean Lab change, not CIEDE2000 and not measured prediction error.
        """
        if (type(max_passes) is not int or not 1 <= max_passes <= 4
                or type(min_passes) is not int or not 2 <= min_passes <= 4
                or isinstance(exit_threshold, (bool, np.bool_))
                or not isinstance(exit_threshold, Real)
                or not np.isfinite(exit_threshold) or exit_threshold < 0
                or (exit_threshold > 0 and min_passes > max_passes)):
            raise ValueError("invalid pass budget or exit policy")
        outputs = []
        reason = "max_passes"
        stream = self.iter_passes(color, patches)
        try:
            for result in stream:
                outputs.append(result)
                if result.index == max_passes:
                    break
                if (exit_threshold > 0 and result.index >= min_passes
                        and result.change_l2 <= exit_threshold):
                    reason = "delta_threshold"
                    break
        finally:
            stream.close()
        return PredictionTrace(tuple(outputs), reason)

    def predict(self, color, patches, **policy):
        return self.predict_trace(color, patches, **policy).prediction
