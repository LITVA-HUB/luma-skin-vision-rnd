"""Matched output-range adapters; the frozen AS implementation stays unchanged.

All heads have value zero and derivative one at the origin. Only the final
residual head changes; recurrent state updates and attention retain AS semantics.
Unit mode delegates directly to AS and preserves its original exported payload.
"""

from __future__ import annotations

import math

import chromaseed_architecture_scale as original
import numpy as np
import torch
from scipy.special import expit
from torch.nn import functional as F

MODES = ("unit", "wide", "linear")


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError("unregistered residual head mode")
    return mode


def head(z, mode):
    validate_mode(mode)
    if mode == "unit":
        return z.tanh()
    if mode == "wide":
        return 4 * (z * 0.25).tanh()
    return z


def numpy_head(z, mode):
    """Independent NumPy expression, used only by the exported CPU consumer."""
    validate_mode(mode)
    if mode == "unit":
        return np.tanh(z)
    if mode == "wide":
        return 4 * np.tanh(z / 4)
    return z


class Bank(original.Bank):
    def __init__(self, variant, mode, seeds=None):
        validate_mode(mode)
        super().__init__(variant, seeds=seeds)
        self.mode = mode

    def forward(self, x, tokens, base):
        if self.mode == "unit":
            return super().forward(x, tokens, base)
        if self.variant == "pool5m":
            t = self.layers["token1"](tokens).relu()
            mean = t.mean(-2)
            spread = ((t - mean.unsqueeze(-2)).square().mean(-2) + 1e-6).sqrt()
            h = self.layers["mlp1"](torch.cat((x, mean, spread, t.amax(-2)), dim=-1)).relu()
            pred = base + head(self.layers["head"](h), self.mode)
            return pred.unsqueeze(-2), pred.sum((1, 2)) * 0.0
        t = F.silu(self.layers["token2"](F.silu(self.layers["token1"](tokens))))
        context_input = torch.cat((x, t.mean(-2)), dim=-1)
        if not self.recurrent:
            h = context_input
            for name in ("mlp1", "mlp2", "mlp3"):
                h = F.silu(self.layers[name](h))
            pred = base + head(self.layers["head"](h), self.mode)
            return pred.unsqueeze(-2), pred.sum((1, 2)) * 0.0
        context = F.silu(self.layers["context"](context_input))
        keys = self.layers["key"](t)
        state, pred = context, base
        outputs, penalties = [], []
        family = "dynamic" if self.variant.startswith("dynamic") else "soft"
        for _ in range(4):
            query = self.layers["query"](torch.cat((state, pred), dim=-1))
            scores = (keys * query.unsqueeze(-2)).sum(-1) / math.sqrt(t.shape[-1])
            attention, _, penalty = original.gate(scores, family, self.training)
            pooled = (t * attention.unsqueeze(-1)).sum(-2)
            u = torch.cat((state, context, pooled, pred), dim=-1)
            update = self.layers["update2"](F.silu(self.layers["update1"](u))).tanh()
            state = 0.5 * state + 0.5 * update
            pred = pred + 0.25 * head(self.layers["head"](state), self.mode)
            outputs.append(pred)
            penalties.append(penalty)
        return torch.stack(outputs, -2), torch.stack(penalties, -1).mean(-1)

    def export(self, slot, warm, tokens):
        model = super().export(slot, warm, tokens)
        if self.mode != "unit":
            model["head_mode"] = np.asarray(self.mode)
        return model


class Predictor(original.Predictor):
    def __init__(self, model):
        self.mode = validate_mode(str(model.get("head_mode", "unit")))
        super().__init__(model)

    def __call__(self, x, tokens, *, all_passes=False):
        if self.mode == "unit":
            return super().__call__(x, tokens, all_passes=all_passes)
        single = np.ndim(x) == 1
        x = np.asarray(x, np.float32).reshape(-1, 36)
        tokens = np.asarray(tokens, np.float32).reshape(-1, 64, 18)
        b = self.base
        xx = ((x - b["x_mean"]) / b["x_std"]).astype(np.float64)
        tt = ((tokens - self.model["t_mean"]) / self.model["t_std"]).astype(np.float64)
        base = np.maximum(xx @ b["w0"].astype(float) + b["b0"], 0) @ b["v0"].astype(float) + b["c0"]

        def layer(name, v):
            w, bias = self.layers[name]
            return v @ w + bias

        def silu(v):
            return v * expit(v)

        def output_head(v):
            return numpy_head(layer("head", v), self.mode)

        if self.variant == "pool5m":
            t = np.maximum(layer("token1", tt), 0)
            mean = t.mean(1)
            spread = np.sqrt(((t - mean[:, None]) ** 2).mean(1) + 1e-6)
            h = np.maximum(layer("mlp1", np.concatenate((xx, mean, spread, t.max(1)), -1)), 0)
            outputs = (base + output_head(h))[:, None]
        else:
            t = silu(layer("token2", silu(layer("token1", tt))))
            ci = np.concatenate((xx, t.mean(1)), -1)
            if self.variant.startswith("patch"):
                h = ci
                for name in ("mlp1", "mlp2", "mlp3"):
                    h = silu(layer(name, h))
                outputs = (base + output_head(h))[:, None]
            else:
                context = silu(layer("context", ci))
                keys = layer("key", t)
                state, pred, outputs = context, base, []
                for _ in range(4):
                    q = layer("query", np.concatenate((state, pred), -1))
                    scores = np.sum(keys * q[:, None], -1) / math.sqrt(t.shape[-1])
                    mask = np.ones_like(scores)
                    if self.variant.startswith("dynamic"):
                        mask = (scores > 0).astype(float)
                        top = np.argsort(scores, axis=-1, kind="stable")[:, -4:]
                        np.put_along_axis(mask, top, 1.0, axis=-1)
                    a = np.exp(scores - scores.max(-1, keepdims=True)) * mask
                    a /= a.sum(-1, keepdims=True)
                    pooled = np.sum(t * a[..., None], 1)
                    u = np.concatenate((state, context, pooled, pred), -1)
                    state = 0.5 * state + 0.5 * np.tanh(layer("update2", silu(layer("update1", u))))
                    pred = pred + 0.25 * output_head(state)
                    outputs.append(pred)
                outputs = np.stack(outputs, 1)
        outputs = outputs * b["y_std"].astype(float) + b["y_mean"].astype(float)
        result = outputs if all_passes else outputs[:, -1]
        return result[0] if single else result


def predict(model, x, tokens):
    consumer = Predictor(model)
    return np.concatenate(
        [consumer(x[i : i + 16], tokens[i : i + 16]) for i in range(0, len(x), 16)]
    )


@torch.no_grad()
def predict_torch(model, x, tokens, device="cuda", all_passes=False):
    mode = validate_mode(str(model.get("head_mode", "unit")))
    if mode == "unit":
        return original.predict_torch(model, x, tokens, device=device, all_passes=all_passes)
    b = original.base_model(model)
    net = Bank(str(model["variant"]), mode, seeds=(17,)).to(device).eval()
    net.theta.copy_(torch.as_tensor(model["theta"], device=device)[None])
    results = []
    for start in range(0, len(x), 64):
        xx = torch.as_tensor((x[start : start + 64] - b["x_mean"]) / b["x_std"], device=device)[
            None
        ]
        tt = torch.as_tensor(
            (tokens[start : start + 64] - model["t_mean"]) / model["t_std"], device=device
        )[None]
        w, bias, v, c = [torch.as_tensor(b[k], device=device) for k in ("w0", "b0", "v0", "c0")]
        base = (xx @ w + bias).relu() @ v + c
        out, _ = net(xx, tt, base)
        native = out[0].cpu().numpy().astype(float) * b["y_std"].astype(float) + b["y_mean"].astype(
            float
        )
        results.append(native if all_passes else native[:, -1])
    return np.concatenate(results)
