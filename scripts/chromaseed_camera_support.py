"""Fixed camera diagnostics and descriptive support; no skin-color model selection."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist

from luma_skin_vision.color import delta_e00

VIEWS = ("color36", "rgb_mean", "lab3", "lab_residual")
METHODS = ("linear", "rbf")
CALIPERS = (1.0, 2.0, 3.0, 5.0, 10.0)


def row_weights(person, site, camera=None):
    person, site = np.asarray(person), np.asarray(site)
    camera = np.zeros(len(person)) if camera is None else np.asarray(camera)
    w = np.zeros(len(person), dtype=np.float64)
    for c in np.unique(camera):
        for p in np.unique(person[camera == c]):
            group = (person == p) & (camera == c)
            for s in np.unique(site[group]):
                mask = group & (site == s)
                w[mask] = 1 / (
                    len(np.unique(camera))
                    * len(np.unique(person[camera == c]))
                    * len(np.unique(site[group]))
                    * mask.sum()
                )
    return w / w.sum()


def aggregate_people(values, person, site):
    return np.array(
        [
            np.mean(
                [
                    np.asarray(values)[(person == p) & (site == s)].mean(axis=0)
                    for s in np.unique(site[person == p])
                ],
                axis=0,
            )
            for p in np.unique(person)
        ]
    )


def standardize(x, q, weights):
    x, q, w = (
        np.asarray(x, dtype=np.float64),
        np.asarray(q, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )
    if (
        x.ndim != 2
        or q.ndim != 2
        or x.shape[1] != q.shape[1]
        or w.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(q).all()
        or not np.isfinite(w).all()
        or np.any(w < 0.0)
        or w.sum() <= 0.0
    ):
        raise ValueError("finite features and nonnegative fit weights with positive mass required")
    w = w / w.sum()
    mean = w @ x
    std = np.maximum(np.sqrt(w @ ((x - mean) ** 2)), 1e-6)
    return (x - mean) / std, (q - mean) / std, dict(mean=mean, std=std)


def views(x, y, qx, qy, w):
    zx, qzx, _ = standardize(x, qx, w)
    zy, qzy, _ = standardize(y, qy, w)
    a, qa = np.column_stack([np.ones(len(y)), zy]), np.column_stack([np.ones(len(qy)), qzy])
    root = np.sqrt(w)
    beta = np.linalg.lstsq(root[:, None] * a, root[:, None] * zx, rcond=1e-12)[0]
    residual, qr, _ = standardize(zx - a @ beta, qzx - qa @ beta, w)
    return (
        dict(color36=zx, rgb_mean=zx[:, 27:30], lab3=zy, lab_residual=residual),
        dict(color36=qzx, rgb_mean=qzx[:, 27:30], lab3=qzy, lab_residual=qr),
    )


def kernel(x, q, width):
    return np.exp(-0.5 * cdist(x, q, "sqeuclidean") / (x.shape[1] * width * width))


def classify(x, q, label, w, method):
    if method == "linear":
        a, b = np.column_stack([np.ones(len(x)), x]), np.column_stack([np.ones(len(q)), q])
        penalty = np.diag(np.r_[0.0, np.full(x.shape[1], 0.1)])
        h, rhs = a.T @ (w[:, None] * a) + penalty, a.T @ (w * label)
        coeff = cho_solve(cho_factor(h, lower=True), rhs)
        return b @ coeff, dict(
            alpha=0.1,
            dimensions=x.shape[1],
            relative_residual=float(
                np.linalg.norm(h @ coeff - rhs) / max(np.linalg.norm(rhs), 1e-15)
            ),
        )
    if method != "rbf":
        raise ValueError("unknown classifier")
    d = np.sqrt(pdist(x, "sqeuclidean") / x.shape[1])
    d = d[d > 1e-10]
    wi = (len(d) - 1) // 2
    width = max(float(np.partition(d, wi)[wi]), 1e-6) if len(d) else 1e-6
    k, root = kernel(x, x, width), np.sqrt(w)
    h, rhs = root[:, None] * k * root[None, :] + 0.01 * np.eye(len(x)), root * label
    coeff = cho_solve(cho_factor(h, lower=True), rhs)
    return kernel(q, x, width) @ (root * coeff), dict(
        alpha=0.01,
        dimensions=x.shape[1],
        width=width,
        relative_residual=float(np.linalg.norm(h @ coeff - rhs) / max(np.linalg.norm(rhs), 1e-15)),
    )


def classification_metrics(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    positive, negative = scores[labels == 1], scores[labels == -1]
    if not len(positive) or not len(negative):
        return None
    auc = np.mean(
        (positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])
    )
    return dict(
        auc=float(auc),
        balanced_accuracy=float(0.5 * (np.mean(positive >= 0) + np.mean(negative < 0))),
        slr_correct=int(np.sum(positive >= 0)),
        slr_people=len(positive),
        ipod_correct=int(np.sum(negative < 0)),
        ipod_people=len(negative),
    )


def match_cost(cost, caliper):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or np.any(cost < 0) or not np.isfinite(cost).all() or caliper < 0:
        raise ValueError("finite nonnegative matching costs and caliper required")
    n, m = cost.shape
    unmatched = (n + 1) * (caliper + 1)
    expanded = np.column_stack(
        [np.where(cost <= caliper, cost, 2 * unmatched), np.full((n, n), unmatched)]
    )
    ri, ci = linear_sum_assignment(expanded)
    valid = ci < m
    return ri[valid], ci[valid]


def nearest(query, fit, kind, query_person=None, fit_person=None):
    out = np.empty(len(query))
    if (query_person is None) != (fit_person is None):
        raise ValueError("both person arrays needed for exclusion")
    for start in range(0, len(query), 48):
        chunk = query[start : start + 48]
        if kind == "rms":
            distance = np.sqrt(cdist(chunk, fit, "sqeuclidean") / fit.shape[1])
        elif kind == "de00":
            distance = delta_e00(chunk[:, None, :], fit[None, :, :])
        else:
            raise ValueError("unknown distance")
        if query_person is not None:
            distance[query_person[start : start + 48, None] == fit_person[None, :]] = np.inf
        out[start : start + 48] = distance.min(axis=1)
    if not np.isfinite(out).all():
        raise ValueError("no eligible neighbor")
    return out


def weighted_quantile(values, weights, fraction):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(np.asarray(weights)[order])
    cumulative /= cumulative[-1]
    return float(
        np.asarray(values)[
            order[min(np.searchsorted(cumulative, fraction, side="left"), len(order) - 1)]
        ]
    )


def describe(values, weights):
    return dict(
        mean=float(np.average(values, weights=weights)),
        median=weighted_quantile(values, weights, 0.5),
        p95=weighted_quantile(values, weights, 0.95),
    )
