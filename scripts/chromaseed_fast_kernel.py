"""Sampled bandwidth and genuinely on-demand Nyström training, CPU float64."""
from __future__ import annotations

import hashlib
import time

import numpy as np
from chromaseed_kernel import (
    EIGEN_FLOOR,
    coordinates,
    fit_normalizer,
    gaussian_kernel,
    median_width,
    nystrom_coefficients,
    select_landmarks,
)

ARMS = ("dense_exact", "column_exact", "column_pairs1024", "column_pairs4096", "column_pairs16384")
RANKS = (64, 128, 256)
ALPHAS = (.1, 1., 10.)
WIDTHS = (.5, 1., 2.)
SEEDS = (17, 29, 43)
PAIR_BUDGETS = (1024, 4096, 16384)
PAIR_SEED_OFFSET = 104729
FIELDS = ("x_mean", "x_std", "y_mean", "y_std", "width", "centers", "coefficient")


def pair_indices(n, count, seed):
    if n < 2 or count < 1:
        raise ValueError("at least two rows and a positive pair count required")
    pairs = np.random.default_rng(seed).integers(0, np.array([n, n - 1]), size=(count, 2), dtype=np.int64)
    pairs[:, 1] += pairs[:, 1] >= pairs[:, 0]
    return pairs


def _positive_median(distance):
    positive = distance[distance > 1e-10]
    if not len(positive):
        return None, 0
    index = (len(positive) - 1) // 2
    return max(float(np.partition(positive, index)[index]), 1e-6), len(positive)


def _sampled_width_bank(x, budgets, seed):
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[1] != 36 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty color36 required")
    if any(p < 1 for p in budgets):
        raise ValueError("positive pair budgets required")
    if len(x) == 1:
        return {p: (1e-6, {"requested_pairs": p, "evaluated_pairs": 0, "positive_pairs": 0, "fallback": "constant", "pair_stream_sha256": None}) for p in budgets}
    pairs = pair_indices(len(x), max(budgets), seed + PAIR_SEED_OFFSET)
    difference = x[pairs[:, 0]] - x[pairs[:, 1]]
    distance = np.sqrt(np.mean(difference * difference, axis=1))
    outputs = {}
    for count in budgets:
        width, positive = _positive_median(distance[:count])
        fallback = "none"
        if width is None:
            anchor = np.sqrt(np.mean((x - x[:1]) ** 2, axis=1))
            width, _ = _positive_median(anchor)
            fallback = "anchor"
            if width is None:
                width, fallback = 1e-6, "constant"
        outputs[count] = (width, {"requested_pairs": count, "evaluated_pairs": count, "positive_pairs": positive,
                                  "fallback": fallback, "pair_stream_sha256": hashlib.sha256(pairs[:count].tobytes()).hexdigest()})
    return outputs


def sampled_width(x, count, seed):
    return _sampled_width_bank(x, (count,), seed)[count]


class KernelColumns:
    """No NxN matrix; count actual queried entries instead of inferred FLOPs."""

    def __init__(self, x, width):
        self.x = np.asarray(x, np.float64)
        if self.x.ndim != 2 or not len(self.x) or self.x.shape[1] != 36 or not np.isfinite(self.x).all():
            raise ValueError("finite color36 required")
        if not np.isfinite(width) or width <= 0:
            raise ValueError("positive finite width required")
        self.squared_norm = np.einsum("ij,ij->i", self.x, self.x)
        self.denominator = 2. * self.x.shape[1] * float(width) ** 2
        self.entries_evaluated = 0

    def column(self, pivot):
        if not 0 <= pivot < len(self.x):
            raise IndexError("pivot outside fit rows")
        distance = np.maximum(self.squared_norm + self.squared_norm[pivot] - 2. * (self.x @ self.x[pivot]), 0.)
        column = np.exp(-distance / self.denominator)
        column[pivot] = 1.
        self.entries_evaluated += len(self.x)
        return column


def streaming_landmarks(operator, weights, count, seed):
    n = len(operator.x)
    weights = np.asarray(weights, np.float64)
    if count < 1 or weights.shape != (n,) or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("positive rank and positive finite row weights required")
    capacity = min(n, count)
    root_w = np.sqrt(weights)
    diagonal = weights.copy()
    tolerance = 1e-12 * max(float(diagonal.max()), 1.)
    # Column-major factors keep every growing prefix contiguous for GEMV.
    factor = np.zeros((n, capacity), np.float64, order="F")
    columns = np.empty((n, capacity), np.float64, order="F")
    rng = np.random.default_rng(seed)
    selected = []
    while len(selected) < capacity and diagonal.max() > tolerance:
        probability = np.where(diagonal > tolerance, diagonal, 0.)
        pivot = int(rng.choice(n, p=probability / probability.sum()))
        j = len(selected)
        raw = operator.column(pivot)
        columns[:, j] = raw
        column = root_w * raw * root_w[pivot] - factor[:, :j] @ factor[pivot, :j]
        factor[:, j] = column / np.sqrt(diagonal[pivot])
        diagonal = np.maximum(diagonal - factor[:, j] ** 2, 0.)
        selected.append(pivot)
        diagonal[selected] = 0.
    k = len(selected)
    info = {"kernel_entries_evaluated": operator.entries_evaluated, "actual_centers": k,
            "largest_training_matrix_shape": [n, capacity],
            "allocated_column_and_factor_bytes": columns.nbytes + factor.nbytes,
            "remaining_weighted_trace": float(diagonal.sum())}
    return np.asarray(selected, np.int64), columns[:, :k], factor[:, :k], diagonal, info


def solve_columns(columns, indices, yn, weights, alphas):
    sub = columns[indices]
    values, vectors = np.linalg.eigh(.5 * (sub + sub.T))
    keep = values > EIGEN_FLOOR * max(float(values.max()), np.finfo(float).tiny)
    if not np.any(keep):
        raise ValueError("no stable landmark direction")
    whitening = vectors[:, keep] / np.sqrt(values[keep])
    design = columns @ whitening
    gram = design.T @ (weights[:, None] * design)
    cross = design.T @ (weights[:, None] * yn)
    eigen, directions = np.linalg.eigh(.5 * (gram + gram.T))
    eigen = np.maximum(eigen, 0.)
    mapped = whitening @ directions
    rhs = directions.T @ cross
    coefficients = []
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError("positive alpha required")
        coefficients.append(mapped @ (rhs / (eigen[:, None] + alpha)))
    return np.stack(coefficients), {"effective_rank": int(keep.sum()), "actual_centers": len(indices),
                                    "smallest_retained_eigenvalue": float(values[keep].min()), "largest_eigenvalue": float(values.max())}


def model_id(arm, rank, seed, wi, ai):
    return f"{arm}_k{rank:03d}_s{seed}_w{wi}_a{ai}"


def payload(prep, normalized, indices, beta, width):
    return {**prep, "centers": normalized[indices].astype(np.float32), "coefficient": beta.astype(np.float32),
            "width": np.asarray(width, np.float32)}


def _landmark_fit(normalized, weights, arm, width, rank, seed):
    if arm == "dense_exact":
        kernel = gaussian_kernel(normalized, normalized, width)
        indices, factor, residual = select_landmarks(kernel, weights, "rpchol", rank, seed)
        info = {"kernel_entries_evaluated": len(normalized) ** 2, "actual_centers": len(indices),
                "largest_training_matrix_shape": list(kernel.shape), "remaining_weighted_trace": float(residual.sum()),
                "kernel_and_factor_bytes": kernel.nbytes + factor.nbytes}
        return indices, kernel, info
    operator = KernelColumns(normalized, width)
    indices, columns, _, _, info = streaming_landmarks(operator, weights, rank, seed)
    return indices, columns, info


def fit_one(x, y, weights, arm, rank, seed, width_index, alpha_index):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    started = time.perf_counter()
    prep = fit_normalizer(x, y)
    normalized = coordinates(prep, x)
    yn = (y.astype(np.float64) - prep["y_mean"]) / prep["y_std"]
    if arm.endswith("exact"):
        base, width_info = median_width(normalized), {"kind": "exact", "requested_pairs": None, "fallback": "none"}
    else:
        base, width_info = sampled_width(normalized, int(arm.removeprefix("column_pairs")), seed)
    width = float(np.float32(base * WIDTHS[width_index]))
    preparation_seconds = time.perf_counter() - started
    ids, design, info = _landmark_fit(normalized, weights, arm, width, rank, seed)
    if arm == "dense_exact":
        beta, solve_info = nystrom_coefficients(design, ids, yn, weights, (ALPHAS[alpha_index],))
    else:
        beta, solve_info = solve_columns(design, ids, yn, weights, (ALPHAS[alpha_index],))
    model = payload(prep, normalized, ids, beta[0], width)
    return model, {**info, "base_width": base, "width_info": width_info, "preparation_seconds": preparation_seconds,
                   "fit_seconds": time.perf_counter() - started, "solve": solve_info, "fit_center_indices": ids.tolist()}


def fit_bank(x, y, weights):
    started = time.perf_counter()
    prep = fit_normalizer(x, y)
    normalized = coordinates(prep, x)
    yn = (y.astype(np.float64) - prep["y_mean"]) / prep["y_std"]
    exact = median_width(normalized)
    sampled = {s: _sampled_width_bank(normalized, PAIR_BUDGETS, s) for s in SEEDS}
    preparation_seconds = time.perf_counter() - started
    models, records, operations = {}, {}, []
    for arm in ARMS:
        for seed in SEEDS:
            if arm.endswith("exact"):
                base, width_info = exact, {"kind": "exact", "requested_pairs": None, "fallback": "none"}
            else:
                base, width_info = sampled[seed][int(arm.removeprefix("column_pairs"))]
            for wi, factor in enumerate(WIDTHS):
                before = time.perf_counter()
                width = float(np.float32(base * factor))
                indices, design, info = _landmark_fit(normalized, weights, arm, width, RANKS[-1], seed)
                operations.append({"arm": arm, "seed": seed, "width_index": wi, "operation": "landmark_fit", **info,
                                   "seconds": time.perf_counter() - before})
                for rank in RANKS:
                    ids = indices[:rank]
                    before = time.perf_counter()
                    if arm == "dense_exact":
                        beta, solve_info = nystrom_coefficients(design, ids, yn, weights, ALPHAS)
                    else:
                        beta, solve_info = solve_columns(design[:, :len(ids)], ids, yn, weights, ALPHAS)
                    operations.append({"arm": arm, "seed": seed, "width_index": wi, "rank": rank, "operation": "readout_bank",
                                       "seconds": time.perf_counter() - before})
                    for ai in range(len(ALPHAS)):
                        key = model_id(arm, rank, seed, wi, ai)
                        models[key] = payload(prep, normalized, ids, beta[ai], width)
                        records[key] = {"arm": arm, "rank": rank, "seed": seed, "width_index": wi, "alpha_index": ai,
                                        "alpha": ALPHAS[ai], "width_factor": factor, "base_width": base, "width_info": width_info,
                                        "fit_center_indices": ids.tolist(), **solve_info}
    return models, {"n_fit_rows": len(x), "readout_configurations": len(models), "preparation_seconds": preparation_seconds,
                    "fit_bank_seconds": time.perf_counter() - started, "operations": operations, "models": records}


def flatten_bank(models):
    return {f"{key}__{field}": array for key, model in models.items() for field, array in model.items()}


def get_model(arrays, key):
    return {field: arrays[f"{key}__{field}"] for field in FIELDS}


def evaluate_bank(models, x, row_indices):
    results = {"row_indices": row_indices}
    for arm in ARMS:
        for seed in SEEDS:
            for wi in range(len(WIDTHS)):
                maximum = models[model_id(arm, RANKS[-1], seed, wi, 0)]
                kernel = gaussian_kernel(coordinates(maximum, x), maximum["centers"], float(maximum["width"]))
                for rank in RANKS:
                    for ai in range(len(ALPHAS)):
                        key = model_id(arm, rank, seed, wi, ai)
                        model = models[key]
                        np.testing.assert_array_equal(model["centers"], maximum["centers"][:len(model["centers"])])
                        results[f"pred__{key}"] = kernel[:, :len(model["centers"])] @ model["coefficient"].astype(np.float64) * model["y_std"] + model["y_mean"]
    return results
