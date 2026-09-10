"""Bounded *hypotheses*, not recovered illuminants. Operate in linear sRGB."""

import numpy as np

from luma_skin_vision.color import linear_to_srgb, srgb_to_linear


def hypotheses(rgb, gain_bounds=(0.8, 1.25)):
    lin = srgb_to_linear(rgb)
    lo, hi = gain_bounds
    if not 0 < lo <= 1 <= hi:
        raise ValueError("gain bounds must contain identity")
    result = {"none": np.asarray(rgb).copy()}
    for name, p in [("gray_world", 1), ("shades_of_gray", 6)]:
        statistic = np.mean(lin**p, axis=(0, 1)) ** (1 / p)
        gain = np.clip(statistic.mean() / np.maximum(statistic, 1e-6), lo, hi)
        result[name] = np.clip(linear_to_srgb(lin * gain), 0, 1)
    return result


def fit_ccm(source_linear, target_linear, *, split, ridge=0.01):
    if split != "train":
        raise ValueError("CCM fitting is restricted to train split")
    x, y = np.asarray(source_linear), np.asarray(target_linear)
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != 3 or len(x) < 4:
        raise ValueError("CCM requires at least four paired RGB triples")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or ridge < 0:
        raise ValueError("invalid CCM values")
    a = np.column_stack([x, np.ones(len(x))])
    return np.linalg.solve(a.T @ a + ridge * np.diag([1, 1, 1, 0]), a.T @ y)


def apply_ccm(source_linear, coefficients):
    x = np.asarray(source_linear)
    return np.concatenate([x, np.ones((*x.shape[:-1], 1))], axis=-1) @ coefficients
