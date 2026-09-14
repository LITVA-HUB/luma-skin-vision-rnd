"""Compact RKHS predictors; FP32 storage and explicit FP64 kernel arithmetic."""

from __future__ import annotations

import numpy as np

EIGEN_FLOOR = 1e-8


def gaussian_kernel(x, centers, width):
    self_kernel = x is centers
    x, centers = np.asarray(x, np.float64), np.asarray(centers, np.float64)
    if x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1] or width <= 0 or not np.isfinite(width):
        raise ValueError("compatible feature matrices and positive width required")
    d2 = np.maximum((x * x).sum(1)[:, None] + (centers * centers).sum(1)[None] - 2. * x @ centers.T, 0.)
    kernel = np.exp(-.5 * d2 / (x.shape[1] * float(width) ** 2))
    if self_kernel:
        np.fill_diagonal(kernel, 1.)
    return kernel


def fit_normalizer(x, y):
    if x.ndim != 2 or x.shape[1] != 36 or y.shape != (len(x), 3) or len(x) == 0:
        raise ValueError("nonempty color36 and native Lab3 required")
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite fit data required")
    return {"x_mean": x.mean(0).astype(np.float32), "x_std": np.maximum(x.std(0), 1e-6).astype(np.float32),
            "y_mean": y.mean(0).astype(np.float32), "y_std": np.maximum(y.std(0), 1e-6).astype(np.float32)}


def coordinates(prep, x):
    return ((np.asarray(x, np.float32) - prep["x_mean"]) / prep["x_std"]).astype(np.float64)


def median_width(x):
    # Direct differences preserve exact duplicates; the squared-distance identity
    # can turn their zero distance into a tiny positive cancellation artifact.
    d2 = np.empty((len(x), len(x)), np.float64)
    for start in range(0, len(x), 32):
        d2[start:start + 32] = ((x[start:start + 32, None] - x[None]) ** 2).sum(-1)
    distance = np.sqrt(d2[np.triu_indices(len(x), 1)] / x.shape[1])
    positive = distance[distance > 1e-10]
    if not len(positive):
        return 1e-6
    index = (len(positive) - 1) // 2
    return max(float(np.partition(positive, index)[index]), 1e-6)


def _eigh(matrix, backend):
    matrix = .5 * (matrix + matrix.T)
    if backend == "cpu":
        return np.linalg.eigh(matrix)
    if backend != "cuda":
        raise ValueError("backend must be cpu or cuda")
    import torch
    values, vectors = torch.linalg.eigh(torch.as_tensor(matrix, dtype=torch.float64, device="cuda"))
    return values.cpu().numpy(), vectors.cpu().numpy()


def exact_coefficients(kernel, y, weights, alphas, backend="cpu"):
    root_w = np.sqrt(weights)
    values, vectors = _eigh(root_w[:, None] * kernel * root_w[None], backend)
    values = np.maximum(values, 0.)
    cross = vectors.T @ (root_w[:, None] * y)
    result = []
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError("positive ridge alpha required")
        result.append(root_w[:, None] * (vectors @ (cross / (values[:, None] + alpha))))
    return np.stack(result)


def select_landmarks(kernel, weights, mode, count, seed=17):
    """Nested pivot selection on sqrt(W) K sqrt(W), without target access."""
    if mode not in ("random", "pivot", "rpchol") or count <= 0:
        raise ValueError("unknown pivot mode or invalid rank")
    weights = np.asarray(weights, np.float64)
    if weights.shape != (len(kernel),) or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("positive finite row weights required")
    root_w = np.sqrt(weights)
    matrix = root_w[:, None] * kernel * root_w[None]
    diagonal = np.diag(matrix).copy()
    tolerance = 1e-12 * max(float(diagonal.max()), 1.)
    factor = np.zeros((len(kernel), min(count, len(kernel))), np.float64)
    rng = np.random.default_rng(seed)
    # Exponential race gives one weighted permutation shared by all rank requests.
    order = np.argsort(-np.log(np.maximum(rng.random(len(kernel)), np.finfo(float).tiny)) / weights)
    if mode == "rpchol":
        rng = np.random.default_rng(seed)
    selected, cursor = [], 0
    while len(selected) < factor.shape[1] and diagonal.max() > tolerance:
        if mode == "random":
            while cursor < len(order) and diagonal[order[cursor]] <= tolerance:
                cursor += 1
            if cursor == len(order):
                break
            pivot = int(order[cursor])
            cursor += 1
        elif mode == "pivot":
            pivot = int(np.argmax(diagonal))
        else:
            probability = np.where(diagonal > tolerance, diagonal, 0.)
            pivot = int(rng.choice(len(kernel), p=probability / probability.sum()))
        j = len(selected)
        column = matrix[:, pivot] - factor[:, :j] @ factor[pivot, :j]
        column = column / np.sqrt(diagonal[pivot])
        factor[:, j] = column
        diagonal = np.maximum(diagonal - column * column, 0.)
        selected.append(pivot)
        diagonal[selected] = 0.
    return np.asarray(selected, np.int64), factor[:, :len(selected)], diagonal


def landmark_whitener(kernel, indices, backend="cpu"):
    submatrix = kernel[np.ix_(indices, indices)]
    values, vectors = _eigh(submatrix, backend)
    keep = values > EIGEN_FLOOR * max(float(values.max()), np.finfo(float).tiny)
    if not np.any(keep):
        raise ValueError("no stable landmark direction")
    whitening = vectors[:, keep] / np.sqrt(values[keep])[None]
    return whitening, {"effective_rank": int(keep.sum()), "landmark_count": len(indices),
                       "smallest_retained_eigenvalue": float(values[keep].min()), "largest_eigenvalue": float(values.max())}


def nystrom_coefficients(kernel, indices, y, weights, alphas, backend="cpu"):
    whitening, info = landmark_whitener(kernel, indices, backend)
    design = kernel[:, indices] @ whitening
    gram = design.T @ (weights[:, None] * design)
    cross = design.T @ (weights[:, None] * y)
    values, vectors = _eigh(gram, backend)
    values = np.maximum(values, 0.)
    mapped = whitening @ vectors
    rhs = vectors.T @ cross
    result = []
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError("positive ridge alpha required")
        result.append(mapped @ (rhs / (values[:, None] + alpha)))
    return np.stack(result), info


def projection_coefficients(kernel, indices, teacher, backend="cpu"):
    whitening, info = landmark_whitener(kernel, indices, backend)
    # Handles one teacher or a leading bank dimension.
    operator = whitening @ (whitening.T @ kernel[indices])
    if teacher.ndim == 2:
        return operator @ teacher, info
    return np.stack([operator @ coefficient for coefficient in teacher]), info


def residual_diagnostic(kernel, indices, teacher, projection):
    """RKHS norm of the ACTUAL (possibly rounded) coefficient difference."""
    residual = np.asarray(teacher, np.float64).copy()
    residual[indices] -= np.asarray(projection, np.float64)
    at_centers = kernel[indices] @ residual
    squared_norm = np.einsum("ic,ic->c", residual, kernel @ residual)
    # In exact arithmetic q >= r(c)^2 for unit-diagonal kernels. Maintain that
    # inequality against small negative cancellation in floating point.
    squared_norm = np.maximum(np.maximum(squared_norm, 0.), (at_centers ** 2).max(0))
    return squared_norm, at_centers


def residual_bound(query_kernel, squared_norm, residual_at_centers):
    """Per-channel bound against the teacher, NOT against instrument truth."""
    nearest = np.argmax(query_kernel, axis=1)
    correlation = np.clip(query_kernel[np.arange(len(query_kernel)), nearest], 0., 1.)[:, None]
    r = residual_at_centers[nearest]
    q = np.asarray(squared_norm, np.float64)[None]
    bound = np.abs(correlation * r) + np.sqrt(np.maximum(q - r * r, 0.)) * np.sqrt(np.maximum(1. - correlation * correlation, 0.))
    return bound + 1e-9 * (1. + np.sqrt(q))


def predict_kernel(payload, x):
    kernel = gaussian_kernel(coordinates(payload, x), payload["centers"], float(payload["width"]))
    return (kernel @ payload["coefficient"].astype(np.float64)) * payload["y_std"] + payload["y_mean"]


def pack_adaptive(prep, centers, levels, tolerance):
    return {**prep, "centers": np.asarray(centers, np.float32),
            "ranks": np.asarray([r for r, _, _, _ in levels], np.int32),
            "coefficients": np.concatenate([b for _, b, _, _ in levels]).astype(np.float32),
            "norm_sq": np.stack([q for _, _, q, _ in levels]).astype(np.float64),
            "residual_values": np.concatenate([v for _, _, _, v in levels]).astype(np.float64),
            "tolerance": np.asarray(tolerance, np.float64)}


class AdaptiveKernel:
    def __init__(self, payload):
        self.prep = payload
        self.centers = payload["centers"].astype(np.float64)
        self.width = float(payload["width"])
        self.ranks = payload["ranks"].astype(int)
        if np.any(np.diff(self.ranks) <= 0) or self.ranks[-1] != len(self.centers):
            raise ValueError("strictly nested center counts ending at payload size required")
        self.tolerance = float(payload["tolerance"])
        self.levels = []
        start = 0
        for i, rank in enumerate(self.ranks):
            self.levels.append((payload["coefficients"][start:start + rank].astype(np.float64),
                                payload["norm_sq"][i], payload["residual_values"][start:start + rank]))
            start += rank
        if start != len(payload["coefficients"]) or start != len(payload["residual_values"]):
            raise ValueError("invalid concatenated prefix payload")

    def predict_one(self, color, tolerance=None):
        tolerance = self.tolerance if tolerance is None else tolerance
        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValueError("finite nonnegative tolerance required")
        x = coordinates(self.prep, np.asarray(color)[None])
        evaluated = []
        previous = 0
        for rank, (coefficient, squared_norm, residual_at_centers) in zip(self.ranks, self.levels, strict=True):
            evaluated.append(gaussian_kernel(x, self.centers[previous:rank], self.width))
            kernel = np.concatenate(evaluated, axis=1)
            prediction = (kernel @ coefficient)[0] * self.prep["y_std"] + self.prep["y_mean"]
            bound = float(np.linalg.norm(residual_bound(kernel, squared_norm, residual_at_centers)[0] * self.prep["y_std"]))
            met = tolerance > 0 and bound <= tolerance
            if met:
                break
            previous = rank
        return prediction, int(rank), bound, bool(met)

    def full_trace(self, x):
        all_kernel = gaussian_kernel(coordinates(self.prep, x), self.centers, self.width)
        outputs, bounds = [], []
        for rank, (coefficient, squared_norm, residual_at_centers) in zip(self.ranks, self.levels, strict=True):
            kernel = all_kernel[:, :rank]
            outputs.append((kernel @ coefficient) * self.prep["y_std"] + self.prep["y_mean"])
            bounds.append(np.linalg.norm(residual_bound(kernel, squared_norm, residual_at_centers) * self.prep["y_std"], axis=1))
        return np.stack(outputs, axis=1), np.stack(bounds, axis=1)


def convex_blend(base, alternative, rho):
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("blend coefficient must lie in [0,1]")
    return (1. - rho) * base + rho * alternative
