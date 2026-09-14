"""Luma ChromaSeed: synthetic color surfaces and a compact transfer model."""
# ruff: noqa: E402
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from luma_skin_vision.color import lab_to_srgb, linear_to_srgb, srgb_to_lab, srgb_to_linear

X_MEAN = np.r_[np.full(30, .5), np.zeros(6)].astype(np.float32)
X_STD = np.r_[np.full(30, .25), np.full(3, .2), np.ones(3)].astype(np.float32)
Y_MEAN = np.array([50, 0, 0], dtype=np.float32)
Y_STD = np.array([25, 30, 30], dtype=np.float32)
QUANTILES = [.01, .05, .1, .25, .5, .75, .9, .95, .99]


def color36(samples):
    """Population statistics matching the real-cache feature definitions."""
    z = np.asarray(samples, dtype=np.float64)
    if z.ndim != 3 or z.shape[2] != 3 or z.shape[1] < 2:
        raise ValueError("Expected batch x samples x RGB")
    if not np.isfinite(z).all() or np.any((z < 0) | (z > 1)):
        raise ValueError("Expected finite encoded RGB in [0,1]")
    mean = z.mean(1)
    centered = z - mean[:, None, :]
    std = z.std(1)
    constant = np.ptp(z, axis=1) == 0
    std[constant] = 0
    covariance = np.einsum("bpc,bpd->bcd", centered, centered) / z.shape[1]
    corr = covariance / np.maximum(std[:, :, None] * std[:, None, :], 1e-12)
    corr = np.where(constant[:, :, None] | constant[:, None, :], 0, corr)
    corr = np.clip(corr, -1, 1)
    quant = np.quantile(z, QUANTILES, axis=1).transpose(1, 0, 2).reshape(len(z), 27)
    return np.column_stack((quant, mean, std, corr[:, 0, 1], corr[:, 0, 2], corr[:, 1, 2]))


def canonical_palette(count, seed):
    """Local D65 colors, sampled without reading any real image or label."""
    if count < 1:
        raise ValueError("Positive palette count required")
    rng = np.random.default_rng(seed)
    colors = []
    accepted = 0
    while accepted < count:
        lab = rng.uniform([10, -80, -80], [95, 80, 90], size=(max(128, (count - accepted) * 4), 3))
        rgb = lab_to_srgb(lab, clip=False)
        good = np.all((rgb >= .02) & (rgb <= .98), axis=1)
        colors.append(rgb[good])
        accepted += int(good.sum())
    rgb = np.concatenate(colors)[:count]
    # Compute from accepted RGB so the canonical label and stored color agree exactly.
    return rgb, srgb_to_lab(rgb)


def render_samples(rgb, seed, style):
    """Approximate camera/illumination randomization, not spectral skin rendering."""
    if style not in ("clean", "rendered"):
        raise ValueError("Unknown renderer")
    base = srgb_to_linear(rgb)
    n, pixels = len(base), 128
    rng = np.random.default_rng(seed)
    shading = rng.normal(size=(n, pixels, 1)) * rng.uniform(0, .04, (n, 1, 1))
    shading -= shading.mean(1, keepdims=True)
    texture = rng.normal(size=(n, pixels, 3)) * rng.uniform(0, .003, (n, 1, 1))
    texture -= texture.mean(1, keepdims=True)
    surface = np.clip(base[:, None, :] * (1 + shading) + texture, 0, 1)
    if style == "rendered":
        exposure = np.exp(rng.uniform(-.45, .45, (n, 1, 1)))
        log_gains = rng.normal(0, .16, (n, 1, 3))
        log_gains -= log_gains.mean(2, keepdims=True)
        surface = surface * exposure * np.exp(log_gains)
        matrix = np.eye(3)[None] + rng.normal(0, .025, (n, 3, 3))
        matrix /= matrix.sum(2, keepdims=True)
        surface = np.einsum("bpc,bdc->bpd", surface, matrix)
        surface += rng.uniform(-.005, .005, (n, 1, 3))
    encoded = linear_to_srgb(np.clip(surface, 0, 1))
    if style == "rendered":
        gamma = np.exp(rng.uniform(-.12, .12, (n, 1, 1)))
        encoded = np.clip(encoded, 0, 1) ** gamma
        saturation = rng.uniform(.85, 1.15, (n, 1, 1))
        neutral = encoded.mean(2, keepdims=True)
        encoded = neutral + saturation * (encoded - neutral)
        encoded += rng.normal(size=encoded.shape) * rng.uniform(0, .003, (n, 1, 1))
    return np.round(np.clip(encoded, 0, 1) * 255) / 255


def make_palette(count, seed):
    rgb, target = canonical_palette(count, seed)
    clean, rendered, visible = [], [], []
    for start in range(0, count, 512):
        part = rgb[start:start + 512]
        render_seed = seed + 104729 * (start // 512 + 1)
        a = render_samples(part, render_seed, "clean")
        b = render_samples(part, render_seed, "rendered")
        clean.append(color36(a).astype(np.float32))
        rendered.append(color36(b).astype(np.float32))
        visible.append(b.mean(1))
    return {"clean": np.concatenate(clean), "rendered": np.concatenate(rendered),
            "target": target, "canonical_rgb": rgb, "observed_rgb": np.concatenate(visible)}


class ChromaSeed(nn.Module):
    """2671 learned scalars. Forward consumes normalized color36, emits normalized Lab."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(36, 64)
        self.out = nn.Linear(64, 3)
        self.skip = nn.Linear(36, 3, bias=False)

    def forward(self, x):
        return self.out(torch.nn.functional.silu(self.hidden(x))) + self.skip(x)


def pack_model(net):
    state = net.state_dict()
    result = {"method": np.asarray("mlp"), "model_name": np.asarray("Luma ChromaSeed v1"),
              "x_mean": X_MEAN.copy(), "x_std": X_STD.copy(),
              "y_mean": Y_MEAN.copy(), "y_std": Y_STD.copy()}
    for key, source in (("hidden_w", "hidden.weight"), ("hidden_b", "hidden.bias"),
                        ("out_w", "out.weight"), ("out_b", "out.bias"), ("skip_w", "skip.weight")):
        result[key] = state[source].detach().cpu().numpy().astype(np.float32).copy()
    return result


def unpack_model(model, device):
    net = ChromaSeed().to(device)
    state = {}
    for key, source in (("hidden_w", "hidden.weight"), ("hidden_b", "hidden.bias"),
                        ("out_w", "out.weight"), ("out_b", "out.bias"), ("skip_w", "skip.weight")):
        state[source] = torch.as_tensor(model[key].copy(), device=device)
    net.load_state_dict(state)
    for key, expected in (("x_mean", X_MEAN), ("x_std", X_STD), ("y_mean", Y_MEAN), ("y_std", Y_STD)):
        np.testing.assert_array_equal(model[key], expected)
    return net


def predict(model, features):
    x = (np.asarray(features, dtype=np.float32) - model["x_mean"]) / model["x_std"]
    h = x @ model["hidden_w"].T + model["hidden_b"]
    h = h / (1 + np.exp(-np.clip(h, -80, 80)))
    y = h @ model["out_w"].T + model["out_b"] + x @ model["skip_w"].T
    return y * model["y_std"] + model["y_mean"]
