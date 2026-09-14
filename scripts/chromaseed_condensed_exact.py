"""Exact lower-median width via compiled condensed distances, no sampling."""
from __future__ import annotations

import time

import numpy as np
from chromaseed_fast_kernel import ALPHAS, WIDTHS, _landmark_fit, payload, solve_columns
from chromaseed_kernel import coordinates, fit_normalizer
from scipy.spatial.distance import pdist


def condensed_width(x):
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[1] != 36 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty color36 required")
    distance = pdist(x, metric="sqeuclidean")
    count, vector_bytes = len(distance), distance.nbytes
    np.divide(distance, 36., out=distance)
    np.sqrt(distance, out=distance)
    positive = distance > 1e-10
    if not positive.all():
        distance = distance[positive]
    if not len(distance):
        width = 1e-6
    else:
        index = (len(distance) - 1) // 2
        distance.partition(index)
        width = max(float(distance[index]), 1e-6)
    return width, {"pair_count": count, "positive_pairs": len(distance), "condensed_distance_vector_bytes": vector_bytes,
                   "kind": "exact_condensed_sqeuclidean", "quadratic_pair_storage": True}


def fit_condensed(x, y, weights, rank, seed, width_index, alpha_index):
    started = time.perf_counter()
    prep = fit_normalizer(x, y)
    normalized = coordinates(prep, x)
    yn = (y.astype(np.float64) - prep["y_mean"]) / prep["y_std"]
    base, width_info = condensed_width(normalized)
    width = float(np.float32(base * WIDTHS[width_index]))
    preparation_seconds = time.perf_counter() - started
    indices, columns, info = _landmark_fit(normalized, weights, "column_exact", width, rank, seed)
    coefficient, solve_info = solve_columns(columns, indices, yn, weights, (ALPHAS[alpha_index],))
    model = payload(prep, normalized, indices, coefficient[0], width)
    return model, {**info, "base_width": base, "width_info": width_info, "solve": solve_info,
                   "preparation_seconds": preparation_seconds, "fit_seconds": time.perf_counter() - started}
