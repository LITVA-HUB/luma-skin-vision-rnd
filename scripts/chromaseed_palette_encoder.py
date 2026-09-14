"""Shared AS-compatible local encoder and normalization-preserving transfer."""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch
from chromaseed_refine import _Layer
from scipy.special import expit
from torch import nn
from torch.nn import functional as F

SEEDS = (17, 29, 43)
SLOTS = tuple((s, a) for s in SEEDS for a in ('aligned', 'shuffled'))
ENCODER_PARAMETERS = 105856
SPECS = (('token1', 18, 384), ('token2', 384, 256), ('palette_aux', 256, 36))
TOTAL_PARAMETERS = 115108


class EncoderBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.empty(6, TOTAL_PARAMETERS))
        self.layers = {}
        offset = 0
        with torch.no_grad():
            for name, a, b in SPECS:
                self.layers[name] = _Layer(self, offset, a, b)
                size = (a+1)*b
                for i, (seed, _) in enumerate(SLOTS):
                    g = torch.Generator().manual_seed(seed+zlib.crc32(name.encode()))
                    self.theta[i, offset:offset+size].uniform_(-1/math.sqrt(a), 1/math.sqrt(a), generator=g)
                offset += size
        assert offset == TOTAL_PARAMETERS

    def encode(self, x):
        return F.silu(self.layers['token2'](F.silu(self.layers['token1'](x))))

    def forward(self, x):
        return self.layers['palette_aux'](self.encode(x))

    def export(self, slot, mean, std):
        return dict(theta=self.theta[slot, :ENCODER_PARAMETERS].detach().numpy().copy(),
                    mean=np.asarray(mean, np.float32).copy(), std=np.asarray(std, np.float32).copy(),
                    seed=np.asarray(SLOTS[slot][0]), arm=np.asarray(SLOTS[slot][1]))


def numpy_encode(model, raw_tokens):
    x = (np.asarray(raw_tokens, np.float64)-model['mean'])/model['std']
    theta = np.asarray(model['theta'], np.float64)
    if theta.shape != (ENCODER_PARAMETERS,) or x.shape[-1] != 18 or not np.isfinite(x).all():
        raise ValueError('Finite raw tokens and complete encoder required')
    offset = 0
    for _, a, b in SPECS[:2]:
        weight = theta[offset:offset+a*b].reshape(a, b)
        offset += a*b
        bias = theta[offset:offset+b]
        offset += b
        x = x@weight+bias
        x = x*expit(x)
    return x


def transplant(model, native_mean, native_std):
    mean, std = np.asarray(native_mean, np.float32), np.asarray(native_std, np.float32)
    if (mean.shape != (18,) or std.shape != (18,) or not (std > 0).all()
            or not np.isfinite(mean).all() or not np.isfinite(std).all()):
        raise ValueError('Finite fit-only mean and positive scale required')
    result = {k: np.array(v, copy=True) for k, v in model.items()}
    theta = model['theta'].astype(np.float64)
    weight = theta[:18*384].reshape(18, 384)
    bias = theta[18*384:19*384]
    new_weight = (std.astype(float)/model['std'])[:, None]*weight
    new_bias = bias+((mean.astype(float)-model['mean'])/model['std'])@weight
    result['theta'][:18*384] = new_weight.ravel()
    result['theta'][18*384:19*384] = new_bias
    result['mean'], result['std'] = mean, std
    return result
