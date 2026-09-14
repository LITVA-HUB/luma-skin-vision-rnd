"""Small independent network banks for the prospective ChromaSeed-R study."""

from __future__ import annotations

import math
import weakref
import zlib

import numpy as np
import torch
from skin_local_search_core import ridge_solve
from torch import nn
from torch.nn import functional as F

FAMILIES = ("stats_mlp", "patch_mlp", "recur_soft", "recur_top16", "recur_dynamic")


def fit_preprocessor(x, patches, y, weights):
    """Caller supplies fit rows only. The native-Lab anchor is not an error head."""
    n = len(x)
    if n == 0 or x.shape != (n, 36) or patches.shape != (n, 64, 18) or y.shape != (n, 3):
        raise ValueError("expected nonempty color36, patches64x18 and native Lab3")
    if weights.shape != (n,) or np.any(weights <= 0):
        raise ValueError("positive weights must match fit rows")
    if not all(np.isfinite(a).all() for a in (x, patches, y, weights)):
        raise ValueError("non-finite fit data")
    xd, td, yd = x.astype(np.float64), patches.astype(np.float64), y.astype(np.float64)
    prep = {"x_mean": xd.mean(0), "x_std": np.maximum(xd.std(0), 1e-6),
            "t_mean": td.mean((0, 1)), "t_std": np.maximum(td.std((0, 1)), 1e-6),
            "y_mean": yd.mean(0), "y_std": np.maximum(yd.std(0), 1e-6)}
    xn, yn = (xd - prep["x_mean"]) / prep["x_std"], (yd - prep["y_mean"]) / prep["y_std"]
    design = torch.from_numpy(np.column_stack((np.ones(n), xn)))
    prep["anchor"] = ridge_solve(design, torch.from_numpy(yn), 1., torch.as_tensor(weights, dtype=torch.float64)).numpy()
    return {key: np.asarray(value, dtype=np.float32) for key, value in prep.items()}


def transform(prep, x, patches):
    xn = (np.asarray(x, np.float32) - prep["x_mean"]) / prep["x_std"]
    tn = (np.asarray(patches, np.float32) - prep["t_mean"]) / prep["t_std"]
    base = xn @ prep["anchor"][1:] + prep["anchor"][0]
    return xn, tn, base


class BankAdamW:
    """AdamW/gradient clipping independently over the leading network dimension.

    A single flat parameter bank avoids one CUDA optimizer launch per small layer.
    Loss must SUM per-network losses, not average them across the bank.
    """

    def __init__(self, parameters, lrs, weight_decay=.01, max_norm=5.):
        self.params = list(parameters)
        if not self.params or any(p.shape[0] != len(lrs) for p in self.params):
            raise ValueError("each parameter must have one leading slot per learning rate")
        self.lrs = torch.as_tensor(lrs, dtype=self.params[0].dtype, device=self.params[0].device)
        if torch.any(self.lrs <= 0):
            raise ValueError("learning rates must be positive")
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.weight_decay, self.max_norm, self.t = weight_decay, max_norm, 0

    @torch.no_grad()
    def step(self, corrections=None):
        if corrections is None:
            self.t += 1
            bias1, bias2_sqrt = 1. - .9 ** self.t, math.sqrt(1. - .999 ** self.t)
        else:
            bias1, bias2_sqrt = corrections[0], corrections[1]
        norm_sq = sum(p.grad.reshape(p.shape[0], -1).square().sum(1)
                      for p in self.params if p.grad is not None)
        coefficient = torch.clamp(self.max_norm / (torch.sqrt(norm_sq) + 1e-6), max=1.)
        for p, m, v in zip(self.params, self.m, self.v, strict=True):
            if p.grad is None:
                continue
            shape = (-1,) + (1,) * (p.ndim - 1)
            grad = p.grad * coefficient.reshape(shape)
            rate = self.lrs.reshape(shape)
            m.lerp_(grad, .1)
            v.mul_(.999).addcmul_(grad, grad, value=.001)
            p.mul_(1. - rate * self.weight_decay)
            denominator = v.sqrt() / bias2_sqrt + 1e-8
            p.add_(-rate / bias1 * m / denominator)


class _Layer:
    """Views are created at each access so no autograd graph survives a step."""

    def __init__(self, net, start, incoming, outgoing, has_bias=True):
        self._net, self.start, self.incoming, self.outgoing = weakref.ref(net), start, incoming, outgoing
        self.has_bias = has_bias

    @property
    def net(self):
        return self._net()

    @property
    def weight(self):
        a, b = self.start, self.start + self.incoming * self.outgoing
        return self.net.theta[:, a:b].reshape(-1, self.incoming, self.outgoing)

    @property
    def bias(self):
        if not self.has_bias:
            raise AttributeError("this projection has no bias")
        a = self.start + self.incoming * self.outgoing
        return self.net.theta[:, a:a + self.outgoing].reshape(-1, 1, self.outgoing)

    def __call__(self, x):
        shape = x.shape
        out = torch.bmm(x.reshape(shape[0], -1, self.incoming), self.weight)
        if self.has_bias:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.outgoing)


def gate_weights(scores, family, training):
    probability = scores.sigmoid()
    if family == "recur_soft":
        mask = torch.ones_like(scores)
    else:
        k = min(16 if family == "recur_top16" else 4, scores.shape[-1])
        fallback = torch.zeros_like(scores).scatter(-1, scores.topk(k, dim=-1).indices, 1.)
        hard = fallback if family == "recur_top16" else torch.maximum(fallback, (scores > 0).to(scores.dtype))
        mask = hard
        if training and family == "recur_dynamic":
            mask = hard + probability - probability.detach()
    count = (mask.detach() > .5).sum(-1).to(scores.dtype)
    unnormalized = torch.exp(scores - scores.amax(-1, keepdim=True)) * mask
    attention = unnormalized / unnormalized.sum(-1, keepdim=True).clamp_min(1e-12)
    penalty = probability.mean(tuple(range(1, probability.ndim))) if family == "recur_dynamic" else probability.sum(tuple(range(1, probability.ndim))) * 0.
    return attention, count, penalty


class BankNet(nn.Module):
    def __init__(self, family, seeds):
        super().__init__()
        if family not in FAMILIES or not seeds:
            raise ValueError("invalid model family or empty bank")
        self.family = family
        self.seeds = tuple(seeds)
        self.recurrent = family.startswith("recur_")
        specs = []
        if family != "stats_mlp":
            specs += [("token1", 18, 32), ("token2", 32, 24)]
        if self.recurrent:
            specs += [("context", 60, 48), ("query", 51, 24), ("key", 24, 24),
                      ("update1", 123, 48), ("update2", 48, 48), ("head", 48, 3)]
        elif family == "patch_mlp":
            specs += [("mlp1", 60, 96), ("mlp2", 96, 64), ("mlp3", 64, 48), ("head", 48, 3)]
        else:
            specs += [("mlp1", 36, 96), ("mlp2", 96, 96), ("mlp3", 96, 48), ("head", 48, 3)]
        # A common key bias cancels in soft attention/top-k, leaving only floating
        # point gradient noise. Omit it consistently in all recurrent controls.
        self.theta = nn.Parameter(torch.empty(len(seeds), sum((a + int(name != "key")) * b for name, a, b in specs)))
        self.layers = {}
        offset = 0
        with torch.no_grad():
            for name, a, b in specs:
                self.layers[name] = _Layer(self, offset, a, b, has_bias=name != "key")
                size = (a + int(name != "key")) * b
                for slot, seed in enumerate(seeds):
                    generator = torch.Generator().manual_seed(seed + zlib.crc32(name.encode()))
                    if name == "head":
                        self.theta[slot, offset:offset + size].zero_()
                    else:
                        self.theta[slot, offset:offset + size].uniform_(-1 / math.sqrt(a), 1 / math.sqrt(a), generator=generator)
                offset += size

    def encode(self, x, patches):
        token = None
        features = x
        if self.family != "stats_mlp":
            token = F.silu(self.layers["token2"](F.silu(self.layers["token1"](patches))))
            features = torch.cat((x, token.mean(-2)), dim=-1)
        if self.recurrent:
            context = F.silu(self.layers["context"](features))
            keys = self.layers["key"](token)
            return token, keys, context
        hidden = features
        for key in ("mlp1", "mlp2", "mlp3"):
            hidden = F.silu(self.layers[key](hidden))
        return hidden, None, None

    def refine(self, state, prediction, token, keys, context, sparse=False):
        query = self.layers["query"](torch.cat((state, prediction), dim=-1))
        scores = (keys * query.unsqueeze(-2)).sum(-1) / math.sqrt(24)
        attention, count, penalty = gate_weights(scores, self.family, self.training)
        if sparse:
            if token.shape[:2] != (1, 1) or self.training:
                raise ValueError("sparse inference is for one image, one model in eval mode")
            active = attention[0, 0] > 0
            pool = (token[:, :, active] * attention[:, :, active, None]).sum(-2)
        else:
            pool = (token * attention.unsqueeze(-1)).sum(-2)
        combined = torch.cat((state, context, pool, prediction), dim=-1)
        update = self.layers["update2"](F.silu(self.layers["update1"](combined))).tanh()
        state = .5 * state + .5 * update
        prediction = prediction + .25 * self.layers["head"](state).tanh()
        return state, prediction, count, penalty

    def forward(self, x, patches, base):
        token, keys, context = self.encode(x, patches)
        if not self.recurrent:
            out = base + self.layers["head"](token).tanh()
            count = torch.full(out.shape[:2] + (1,), 0. if self.family == "stats_mlp" else float(patches.shape[-2]), device=out.device, dtype=out.dtype)
            return out.unsqueeze(-2), count, out.sum((1, 2)) * 0.
        state, prediction = context, base
        outputs, counts, penalties = [], [], []
        for _ in range(4):
            state, prediction, count, penalty = self.refine(state, prediction, token, keys, context)
            outputs.append(prediction)
            counts.append(count)
            penalties.append(penalty)
        return torch.stack(outputs, dim=-2), torch.stack(counts, dim=-1), torch.stack(penalties, dim=-1).mean(-1)

    def export_slot(self, slot):
        return {"family": np.asarray(self.family), "theta": self.theta[slot].detach().cpu().numpy().astype(np.float32).copy()}

    @classmethod
    def from_payload(cls, payload):
        net = cls(str(payload["family"]), [17])
        with torch.no_grad():
            net.theta.copy_(torch.from_numpy(payload["theta"])[None])
        return net


def apply_exit_policy(outputs_lab, threshold):
    """Offline equivalent of actual inference stopping; native Lab distance."""
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and nonnegative")
    if outputs_lab.ndim != 3 or outputs_lab.shape[-1] != 3:
        raise ValueError("expected rows x passes x Lab3")
    n, passes = outputs_lab.shape[:2]
    steps = np.full(n, passes, dtype=np.int64)
    if threshold > 0 and passes > 1:
        change = np.linalg.norm(np.diff(outputs_lab, axis=1), axis=-1)
        for index in range(1, passes):
            stop = (steps == passes) & (change[:, index - 1] <= threshold)
            steps[stop] = index + 1
    return outputs_lab[np.arange(n), steps - 1], steps


@torch.no_grad()
def predict_one(net, x, patches, base, y_std, threshold=0., sparse=False):
    """Actually execute only required passes, for batch-one CPU deployment timing."""
    if net.training or x.shape[:2] != (1, 1) or net.theta.shape[0] != 1:
        raise ValueError("one input and one eval-mode model are required")
    token, keys, context = net.encode(x, patches)
    if not net.recurrent:
        prediction = base + net.layers["head"](token).tanh()
        return prediction, 1, [0 if net.family == "stats_mlp" else patches.shape[-2]]
    scale = torch.as_tensor(y_std, device=x.device, dtype=x.dtype)
    state, prediction, counts = context, base, []
    for index in range(4):
        previous = prediction
        state, prediction, count, _ = net.refine(state, prediction, token, keys, context, sparse=sparse)
        counts.append(float(count.item()))
        if threshold > 0 and index >= 1 and torch.linalg.vector_norm((prediction - previous) * scale).item() <= threshold:
            break
    return prediction, index + 1, counts
