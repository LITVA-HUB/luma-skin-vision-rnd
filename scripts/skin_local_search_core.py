"""Float64 closed-form primitives for bounded local skin-model searches."""

import math

import torch


def _require_float64(*values):
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if any(value.dtype != torch.float64 for value in tensors):
        raise ValueError("all tensor inputs must use torch.float64")
    if tensors and any(value.device != tensors[0].device for value in tensors[1:]):
        raise ValueError("all tensor inputs must use the same device")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("tensor inputs must be finite")


def _positive_scalar(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _validated_weights(weights, rows, reference):
    if weights is None:
        return torch.ones(rows, dtype=torch.float64, device=reference.device)
    _require_float64(weights)
    if weights.device != reference.device or weights.shape != (rows,):
        raise ValueError("weights must match rows and device")
    if torch.any(weights <= 0):
        raise ValueError("weights must be strictly positive")
    return weights


def _regression_inputs(design, target, alpha, weights):
    _require_float64(design, target)
    alpha = _positive_scalar(alpha, "alpha")
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] == 0:
        raise ValueError("design must be a nonempty matrix")
    if target.ndim not in (1, 2) or target.shape[0] != design.shape[0]:
        raise ValueError("target must match design rows")
    if target.device != design.device:
        raise ValueError("target must use the design device")
    return alpha, _validated_weights(weights, design.shape[0], design)


def ridge_solve(design, y, alpha, weights=None):
    """Solve weighted ridge regression, leaving design column zero unpenalized."""
    alpha, weight = _regression_inputs(design, y, alpha, weights)
    system = design.T @ (weight[:, None] * design)
    penalty = torch.eye(design.shape[1], dtype=torch.float64, device=design.device) * alpha
    penalty[0, 0] = 0.0
    rhs = design.T @ (weight[:, None] * y if y.ndim == 2 else weight * y)
    return torch.linalg.solve(system + penalty, rhs)


def regularized_objective(design, y, beta, alpha, weights=None):
    """Return weighted squared error plus ridge penalty, excluding row zero of beta."""
    alpha, weight = _regression_inputs(design, y, alpha, weights)
    _require_float64(beta)
    if beta.device != design.device or beta.shape != (design.shape[1], *y.shape[1:]):
        raise ValueError("beta shape or device does not match regression inputs")
    residual = y - design @ beta
    error = torch.sum(weight[:, None] * residual.square()) if y.ndim == 2 else torch.sum(weight * residual.square())
    return error + alpha * torch.sum(beta[1:].square())


def rbf_features(x, centers, widths):
    """Evaluate isotropic Gaussian features using mean squared coordinate distance."""
    width = widths if isinstance(widths, torch.Tensor) else torch.tensor(widths, dtype=torch.float64, device=x.device)
    _require_float64(x, centers, width)
    if x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1]:
        raise ValueError("x and centers must be matrices with equal feature width")
    if centers.device != x.device or width.device != x.device:
        raise ValueError("x, centers and widths must use the same device")
    if width.ndim == 0:
        width = width.expand(centers.shape[0])
    if width.shape != (centers.shape[0],):
        raise ValueError("widths must be scalar or one value per center")
    if torch.any(width <= 0):
        raise ValueError("widths must be strictly positive")
    distance = torch.mean((x[:, None, :] - centers[None, :, :]).square(), dim=2)
    return torch.exp(-0.5 * distance / width[None, :].square())


def greedy_ridge_indices(base, z, y, alpha, count, weights=None):
    """Select ridge atoms by exact Schur-complement objective reductions."""
    alpha, weight = _regression_inputs(base, y, alpha, weights)
    _require_float64(z)
    if z.ndim != 2 or z.shape[0] != base.shape[0] or z.device != base.device:
        raise ValueError("z must match base rows and device")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if z.shape[1] == 0 or count == 0:
        return [], []

    target = y[:, None] if y.ndim == 1 else y
    weighted_base = weight[:, None] * base
    weighted_z = weight[:, None] * z
    base_system = base.T @ weighted_base
    penalty = torch.eye(base.shape[1], dtype=torch.float64, device=base.device) * alpha
    penalty[0, 0] = 0.0
    base_system = base_system + penalty
    cross = base.T @ weighted_z
    base_rhs = base.T @ (weight[:, None] * target)
    beta = torch.linalg.solve(base_system, base_rhs)

    schur = z.T @ weighted_z + alpha * torch.eye(z.shape[1], dtype=torch.float64, device=z.device)
    schur = schur - cross.T @ torch.linalg.solve(base_system, cross)
    error_correlation = z.T @ (weight[:, None] * target) - cross.T @ beta
    schur = 0.5 * (schur + schur.T)

    active = torch.ones(z.shape[1], dtype=torch.bool, device=z.device)
    indices = []
    gains = []
    for _ in range(min(count, z.shape[1])):
        diagonal = torch.diagonal(schur)
        if torch.any(~torch.isfinite(diagonal[active])) or torch.any(diagonal[active] <= 0):
            raise ValueError("nonpositive or nonfinite Schur complement")
        score = torch.sum(error_correlation.square(), dim=1) / diagonal
        score = torch.where(active, score, torch.full_like(score, -torch.inf))
        index = int(torch.argmax(score).item())
        denominator = schur[index, index].clone()
        column = schur[:, index].clone()
        pivot_error = error_correlation[index].clone()
        indices.append(index)
        gains.append(float(score[index].item()))
        error_correlation = error_correlation - column[:, None] * (pivot_error / denominator)
        schur = schur - torch.outer(column, column) / denominator
        schur = 0.5 * (schur + schur.T)
        active[index] = False
    return indices, gains


def kernel_ridge_fit(x, y, width, alpha, weights=None):
    """Fit dual coefficients for weighted Gaussian kernel ridge without a bias."""
    _require_float64(x, y)
    alpha = _positive_scalar(alpha, "alpha")
    width = _positive_scalar(width, "width")
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must be a nonempty matrix")
    if y.ndim not in (1, 2) or y.shape[0] != x.shape[0] or y.device != x.device:
        raise ValueError("y must match x rows and device")
    weight = _validated_weights(weights, x.shape[0], x)
    kernel = rbf_features(x, x, width)
    system = kernel + torch.diag(alpha / weight)
    return torch.linalg.solve(system, y)


def kernel_ridge_predict(query, x, coeff, width):
    """Predict from Gaussian kernel ridge coefficients."""
    _require_float64(query, x, coeff)
    _positive_scalar(width, "width")
    if query.ndim != 2 or x.ndim != 2 or query.shape[1] != x.shape[1]:
        raise ValueError("query and x must have equal feature width")
    if coeff.ndim not in (1, 2) or coeff.shape[0] != x.shape[0]:
        raise ValueError("coeff must match x rows")
    if query.device != x.device or coeff.device != x.device:
        raise ValueError("query, x and coeff must use the same device")
    return rbf_features(query, x, width) @ coeff
