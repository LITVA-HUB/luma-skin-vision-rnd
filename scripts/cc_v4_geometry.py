"""Grounded correction-action targets, not a learned model or novel theorem.

An action is log(R/G),log(B/G) of a candidate illuminant; it implies a
positive diagonal correction. Targets follow a supplied real illuminant GT.
They measure neutral reproduction, never physical surface DeltaE. Multiple
queries reuse one GT and do not create independent labeled scenes.
"""

import numpy as np


def correction_targets(gt, actions):
    """Return per-action angle, smooth sin²(angle) and analytic action gradient.

    gt: Nx3 positive finite camera RGB. actions: NxKx2 finite log ratios,
    bounded to absolute value30 to keep the declared numerical domain finite.
    A joint diagonal gain shifts GT and action log ratios by the same amount.
    That identity is not a claim for arbitrary unknown nonlinear image ISPs.
    """
    gt, actions = np.asarray(gt, dtype=np.float64), np.asarray(actions, dtype=np.float64)
    if (
        gt.ndim != 2
        or gt.shape[-1] != 3
        or len(gt) == 0
        or actions.ndim != 3
        or actions.shape[0] != len(gt)
        or actions.shape[1] == 0
        or actions.shape[-1] != 2
        or not np.isfinite(gt).all()
        or np.any(gt <= 0)
        or not np.isfinite(actions).all()
        or np.any(np.abs(actions) > 30)
    ):
        raise ValueError("Expected positive finite Nx3 GT and bounded finite NxKx2 actions")
    log_candidate = np.stack((actions[..., 0], np.zeros(actions.shape[:2]), actions[..., 1]), -1)
    residual = np.log(gt[:, None]) - log_candidate
    residual -= residual.max(-1, keepdims=True)
    ratio = np.exp(residual)
    simplex = ratio / ratio.sum(-1, keepdims=True)
    second = np.square(simplex).sum(-1)
    cost = np.clip(1 - 1 / (3 * second), 0, 2 / 3)
    normalized = simplex[..., [0, 2]] / second[..., None]
    gradient = (2 / 3) * (normalized - np.square(normalized))
    neutral = np.ones_like(ratio) / np.sqrt(3)
    angle = np.degrees(
        np.arctan2(np.linalg.norm(np.cross(ratio, neutral), axis=-1), (ratio * neutral).sum(-1))
    )
    return {"sin2_cost": cost, "cost_gradient": gradient, "reproduction_degrees": angle}
