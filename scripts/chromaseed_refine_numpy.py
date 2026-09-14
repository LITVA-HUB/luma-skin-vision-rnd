"""Independent NumPy batch-one deployment path, including actual early stopping."""

from __future__ import annotations

import numpy as np


def silu(x):
    return x / (1. + np.exp(-np.clip(x, -80., 80.)))


class NumpyRefiner:
    def __init__(self, payload):
        self.family = str(payload["family"])
        self.recurrent = self.family.startswith("recur_")
        self.prep = {k: np.asarray(payload[k], np.float32) for k in ("x_mean", "x_std", "t_mean", "t_std", "y_mean", "y_std", "anchor")}
        theta = np.asarray(payload["theta"], np.float32)
        self.threshold = float(payload.get("exit_threshold", 0.))
        if theta.ndim != 1:
            raise ValueError("deployment payload must contain exactly one model")
        specs = []
        if self.family != "stats_mlp":
            specs += [("token1", 18, 32), ("token2", 32, 24)]
        if self.recurrent:
            specs += [("context", 60, 48), ("query", 51, 24), ("key", 24, 24),
                      ("update1", 123, 48), ("update2", 48, 48), ("head", 48, 3)]
        elif self.family == "patch_mlp":
            specs += [("mlp1", 60, 96), ("mlp2", 96, 64), ("mlp3", 64, 48), ("head", 48, 3)]
        elif self.family == "stats_mlp":
            specs += [("mlp1", 36, 96), ("mlp2", 96, 96), ("mlp3", 96, 48), ("head", 48, 3)]
        else:
            raise ValueError("unknown family")
        self.layers = {}
        offset = 0
        for name, incoming, outgoing in specs:
            size = incoming * outgoing
            weight = theta[offset:offset + size].reshape(incoming, outgoing)
            offset += size
            bias = None
            if name != "key":
                bias = theta[offset:offset + outgoing]
                offset += outgoing
            self.layers[name] = weight, bias
        if offset != len(theta):
            raise ValueError("payload length does not match architecture")

    def layer(self, name, value):
        weight, bias = self.layers[name]
        projected = value @ weight
        return projected if bias is None else projected + bias

    def predict_trace(self, color, patches, threshold=None, sparse=True):
        """Runs one skin region. No torch, image extraction or face detection."""
        threshold = self.threshold if threshold is None else threshold
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("invalid exit threshold")
        p = self.prep
        x = (np.asarray(color, np.float32) - p["x_mean"]) / p["x_std"]
        prediction = x @ p["anchor"][1:] + p["anchor"][0]
        features = x
        if self.family != "stats_mlp":
            t = (np.asarray(patches, np.float32) - p["t_mean"]) / p["t_std"]
            token = silu(self.layer("token2", silu(self.layer("token1", t))))
            features = np.concatenate((x, token.mean(0)))
        if not self.recurrent:
            hidden = features
            for name in ("mlp1", "mlp2", "mlp3"):
                hidden = silu(self.layer(name, hidden))
            output = (prediction + np.tanh(self.layer("head", hidden))) * p["y_std"] + p["y_mean"]
            return output[None], np.array([0 if self.family == "stats_mlp" else len(patches)], np.int64)
        context = silu(self.layer("context", features))
        state = context
        keys = self.layer("key", token)
        outputs, connections = [], []
        for index in range(4):
            previous = prediction
            query = self.layer("query", np.concatenate((state, prediction)))
            scores = (keys * query).sum(-1) / np.float32(np.sqrt(24.))
            active = np.ones(len(scores), bool)
            if self.family == "recur_top16":
                active[:] = False
                active[np.argpartition(scores, -16)[-16:]] = True
            elif self.family == "recur_dynamic":
                active = scores > 0
                active[np.argpartition(scores, -4)[-4:]] = True
            attention = np.exp(scores - scores.max()) * active.astype(np.float32)
            attention /= attention.sum()
            if sparse:
                pool = (token[active] * attention[active, None]).sum(0)
            else:
                pool = (token * attention[:, None]).sum(0)
            combined = np.concatenate((state, context, pool, prediction))
            update = np.tanh(self.layer("update2", silu(self.layer("update1", combined))))
            state = .5 * state + .5 * update
            prediction = prediction + .25 * np.tanh(self.layer("head", state))
            outputs.append(prediction * p["y_std"] + p["y_mean"])
            connections.append(int(active.sum()))
            if threshold > 0 and index >= 1 and np.linalg.norm((prediction - previous) * p["y_std"]) <= threshold:
                break
        return np.stack(outputs), np.asarray(connections)

    def predict(self, color, patches, threshold=None, sparse=True):
        outputs, counts = self.predict_trace(color, patches, threshold, sparse)
        return outputs[-1], len(outputs), counts
