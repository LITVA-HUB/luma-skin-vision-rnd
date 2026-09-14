"""Effective readout capacity with an explicitly unpenalized intercept."""

from __future__ import annotations

import numpy as np
from chromaseed_perceptual import local_tensor


def source_design(source, x):
    x = np.asarray(x, np.float32)
    selected = x[:, source["feature_indices"]] if "feature_indices" in source else x
    xn = ((selected - source["x_mean"]) / source["x_std"]).astype(np.float64)
    h = np.maximum(xn @ source["w1"].astype(float).T + source["b1"].astype(float), 0)
    raw_std = h.std(0)
    z = np.column_stack((np.ones(len(x)), (h - h.mean(0)) / np.maximum(raw_std, 1e-6)))
    return z, int(np.sum(raw_std < 1e-6))


def source_metric(source, y, weights):
    tensor, _ = local_tensor(np.asarray(y, np.float64))
    ys = source["y_std"].astype(float)
    m = tensor * ys[None, :, None] * ys[None, None, :]
    scale = np.average(np.trace(m, axis1=1, axis2=2), weights=weights) / 3
    return m / scale


def spectrum(z, weights, metric=None):
    z, w = np.asarray(z, float), np.asarray(weights, float)
    if (
        z.ndim != 2
        or not len(z)
        or z.shape[1] < 1
        or w.shape != (len(z),)
        or not np.isfinite(z).all()
        or not np.isfinite(w).all()
        or np.any(w <= 0)
        or not np.array_equal(z[:, 0], np.ones(len(z)))
    ):
        raise ValueError("finite design, intercept and positive weights required")
    width = z.shape[1]
    if metric is None:
        gram = z.T @ (w[:, None] * z)
        bias, multiplicity = 1, 3
    else:
        m = np.asarray(metric, float)
        if (
            m.shape != (len(z), 3, 3)
            or not np.isfinite(m).all()
            or not np.allclose(m, m.transpose(0, 2, 1), atol=1e-12)
            or np.min(np.linalg.eigvalsh(m)) <= 0
        ):
            raise ValueError("finite SPD per-row metric required")
        gram = np.empty((width * 3, width * 3))
        for a in range(3):
            for b in range(a, 3):
                block = z.T @ ((w * m[:, a, b])[:, None] * z)
                gram[a::3, b::3], gram[b::3, a::3] = block, block.T
        bias, multiplicity = 3, 1
    cross = gram[:bias, bias:]
    schur = gram[bias:, bias:] - cross.T @ np.linalg.solve(gram[:bias, :bias], cross)
    eigenvalues = np.linalg.eigvalsh((schur + schur.T) * 0.5)
    tolerance = 1e-8 * max(1.0, float(np.max(np.abs(gram))))
    if len(eigenvalues) and eigenvalues[0] < -tolerance:
        raise FloatingPointError("non-PSD residual Gram beyond roundoff")
    return np.maximum(eigenvalues, 0)[::-1], bias, multiplicity


def degrees(value, alpha):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("positive finite penalty required")
    eigenvalues, bias, multiplicity = value
    return float(multiplicity * (bias + np.sum(eigenvalues / (eigenvalues + alpha))))
