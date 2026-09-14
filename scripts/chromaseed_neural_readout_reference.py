"""Independent hidden design, analytic perceptual metric and augmented pivoted QR."""

from __future__ import annotations

import numpy as np
from chromaseed_perceptual_reference import analytic_tensor
from scipy.linalg import lstsq


def refit(source, x, y, weights, family, alpha):
    coordinates = source.get("feature_indices", np.arange(36))
    z = ((x[:, coordinates].astype(np.float32) - source["x_mean"]) / source["x_std"]).astype(
        np.float64
    )
    activation = np.clip(
        z.dot(source["w1"].astype(np.float64).T) + source["b1"].astype(np.float64), 0, None
    )
    mu = np.sum(activation, axis=0) / len(activation)
    sd = np.maximum(np.sqrt(np.sum(np.square(activation - mu), axis=0) / len(activation)), 1e-6)
    features = np.concatenate((np.ones((len(x), 1)), (activation - mu) / sd), axis=1)
    target = (np.asarray(y, np.float64) - source["y_mean"]) / source["y_std"]
    weights = np.asarray(weights, np.float64)
    n, width = features.shape
    scale = None
    if family == "norm":
        design = features * np.sqrt(weights)[:, None]
        rhs = target * np.sqrt(weights)[:, None]
        penalty = np.diag(np.r_[0.0, np.full(width - 1, np.sqrt(alpha))])
        design = np.concatenate((design, penalty), axis=0)
        rhs = np.concatenate((rhs, np.zeros((width, 3))), axis=0)
    elif family == "perceptual":
        metric = analytic_tensor(y)
        metric = (
            metric
            * source["y_std"].astype(float)[None, :, None]
            * source["y_std"].astype(float)[None, None, :]
        )
        scale = float(np.sum(weights * np.trace(metric, axis1=1, axis2=2)) / np.sum(weights) / 3)
        left = (
            np.linalg.cholesky(metric / scale).transpose(0, 2, 1) * np.sqrt(weights)[:, None, None]
        )
        design = np.empty((n * 3, width * 3))
        for i in range(n):
            design[3 * i : 3 * i + 3] = np.kron(features[i : i + 1], left[i])
        rhs = np.einsum("nij,nj->ni", left, target).ravel()
        penalty = np.diag(np.r_[np.zeros(3), np.full((width - 1) * 3, np.sqrt(alpha))])
        design = np.concatenate((design, penalty), axis=0)
        rhs = np.concatenate((rhs, np.zeros(width * 3)))
    else:
        raise ValueError("registered output objective required")
    solution, _, rank, _ = lstsq(design, rhs, cond=None, lapack_driver="gelsy", check_finite=True)
    beta = solution.reshape(width, 3)
    coefficients = beta[1:] / sd[:, None]
    result = {k: v.copy() for k, v in source.items()}
    result["w2"] = np.asarray(coefficients.T, np.float32)
    result["b2"] = np.asarray(beta[0] - np.sum(mu[:, None] * coefficients, axis=0), np.float32)
    return result, dict(
        rank=int(rank), metric_scale=scale, design_rows=len(design), design_columns=design.shape[1]
    )
