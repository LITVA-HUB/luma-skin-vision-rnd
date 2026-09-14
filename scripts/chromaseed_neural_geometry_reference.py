"""Independent whitened-design QR/SVD and analytic perceptual metric."""

from __future__ import annotations

import numpy as np
from chromaseed_perceptual_reference import analytic_tensor
from scipy.linalg import qr, svdvals


def reconstruct(source, x, y, weights, family):
    coords = source.get("feature_indices", np.arange(36))
    normalized = ((x[:, coords].astype(np.float32) - source["x_mean"]) / source["x_std"]).astype(
        float
    )
    activation = np.clip(
        normalized.dot(source["w1"].astype(float).T) + source["b1"].astype(float), 0, None
    )
    mean = np.sum(activation, axis=0) / len(x)
    std = np.sqrt(np.sum((activation - mean) ** 2, axis=0) / len(x))
    design = np.concatenate(
        (np.ones((len(x), 1)), (activation - mean) / np.maximum(std, 1e-6)), axis=1
    )
    if family == "norm":
        whitened = np.sqrt(weights)[:, None] * design
        bias, multiplicity = 1, 3
    elif family == "perceptual":
        tensor = analytic_tensor(y)
        ys = source["y_std"].astype(float)
        metric = tensor * ys[None, :, None] * ys[None, None, :]
        scale = np.sum(weights * np.trace(metric, axis1=1, axis2=2)) / np.sum(weights) / 3
        left = (
            np.linalg.cholesky(metric / scale).transpose(0, 2, 1) * np.sqrt(weights)[:, None, None]
        )
        whitened = np.einsum("ni,nab->naib", design, left).reshape(3 * len(x), 3 * design.shape[1])
        bias, multiplicity = 3, 1
    else:
        raise ValueError("registered objective required")
    q, _ = qr(whitened[:, :bias], mode="economic")
    remainder = whitened[:, bias:] - q @ (q.T @ whitened[:, bias:])
    values = svdvals(remainder) ** 2
    return values, bias, multiplicity
