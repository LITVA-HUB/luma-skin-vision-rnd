"""Small independent continuous-Lab denoising blocks (ND)."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

SPECS = {
    "plain": (1, 36, 69),
    "local2": (2, 39, 32),
    "local4": (4, 39, 16),
    "blind4": (4, 39, 16),
    "e2e4": (4, 39, 16),
}


def schedule(k):
    if k not in (1, 2, 4):
        raise ValueError("unsupported step count")
    angle = np.arange(k + 1, dtype=np.float64) * np.pi / (2 * k)
    a, s = np.sin(angle), np.cos(angle)
    a[0], s[0], a[-1], s[-1] = 0.0, 1.0, 1.0, 0.0
    return a, s


def preprocessor(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim != 2 or x.shape[1] != 36 or not len(x) or y.shape != (len(x), 3):
        raise ValueError("nonempty color36 and matching Lab3 required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite fitting data")
    return {
        key: value.astype(np.float32)
        for key, value in {
            "x_mean": x.mean(0),
            "x_std": np.maximum(x.std(0), 1e-6),
            "y_mean": y.mean(0),
            "y_std": np.maximum(y.std(0), 1e-6),
        }.items()
    }


class Bank(nn.Module):
    """Leading optimizer slots are individual blocks, never shared moments."""

    def __init__(self, family, seeds):
        super().__init__()
        if family not in SPECS or not seeds:
            raise ValueError("unknown family or empty slots")
        self.family, self.m = family, len(seeds)
        self.k, self.d, self.h = SPECS[family]
        self.p = (self.d + 1) * self.h + (self.h + 1) * 3
        initial = np.zeros((self.m, self.k, self.p), np.float32)
        for i, seed in enumerate(seeds):
            for b in range(self.k):
                rng = np.random.default_rng(int(seed) + 310003 + 1009 * b)
                initial[i, b, : self.d * self.h] = rng.uniform(
                    -1 / math.sqrt(self.d), 1 / math.sqrt(self.d), self.d * self.h
                )
                start = (self.d + 1) * self.h
                initial[i, b, start : start + self.h * 3] = rng.uniform(
                    -1 / math.sqrt(self.h), 1 / math.sqrt(self.h), self.h * 3
                )
        self.theta = nn.Parameter(torch.from_numpy(initial.reshape(self.m * self.k, self.p)))
        self.a, self.s = schedule(self.k)

    def layer(self, incoming, theta):
        # incoming G x N x D, theta G x P; G can be models or models*blocks.
        end = self.d * self.h
        hidden = torch.relu(
            torch.bmm(incoming, theta[:, :end].reshape(-1, self.d, self.h))
            + theta[:, end : end + self.h, None].transpose(1, 2)
        )
        start = end + self.h
        return torch.bmm(
            hidden, theta[:, start : start + 3 * self.h].reshape(-1, self.h, 3)
        ) + theta[:, -3:, None].transpose(1, 2)

    def local(self, x, state):
        n = x.shape[1]
        expanded = x[:, None].expand(self.m, self.k, n, 36)
        if self.family == "plain":
            incoming = expanded
        else:
            if self.family == "blind4":
                state = torch.zeros_like(state)
            incoming = torch.cat((expanded, state), -1)
        return self.layer(incoming.reshape(self.m * self.k, n, self.d), self.theta).reshape(
            self.m, self.k, n, 3
        )

    def rollout(self, x):
        state = torch.zeros((*x.shape[:2], 3), dtype=x.dtype, device=x.device)
        theta = self.theta.reshape(self.m, self.k, self.p)
        predictions = []
        for t in range(self.k):
            visible = torch.zeros_like(state) if self.family == "blind4" else state
            incoming = x if self.family == "plain" else torch.cat((x, visible), -1)
            clean = self.layer(incoming, theta[:, t])
            predictions.append(clean)
            state = self.a[t + 1] * clean + self.s[t + 1] / self.s[t] * (state - self.a[t] * clean)
        return torch.stack(predictions, dim=1)

    def export(self, slot, prep):
        if not 0 <= slot < self.m:
            raise ValueError("invalid model slot")
        return {
            **{k: v.copy() for k, v in prep.items()},
            "family": np.asarray(self.family),
            "theta": self.theta.detach().reshape(self.m, self.k, self.p)[slot].cpu().numpy().copy(),
        }
