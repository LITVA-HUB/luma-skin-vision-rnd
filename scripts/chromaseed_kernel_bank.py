"""Reusable algebra banks for ChromaSeed-K; fit and query labels are separate."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_kernel import (
    coordinates,
    exact_coefficients,
    fit_normalizer,
    gaussian_kernel,
    median_width,
    nystrom_coefficients,
    pack_adaptive,
    projection_coefficients,
    residual_bound,
    residual_diagnostic,
    select_landmarks,
)

ALPHAS = (.1, 1., 10.)
WIDTHS = (.5, 1., 2.)
RANKS = (16, 32, 64, 128)
SEEDS = (17, 29, 43)
FAMILIES = ("exact", "nys_random", "nys_pivot", "nys_rpchol", "project_rpchol")
MODEL_FIELDS = ("x_mean", "x_std", "y_mean", "y_std", "width", "centers", "coefficient")


def seeds_for(family):
    return (17,) if family in ("exact", "nys_pivot") else SEEDS


def model_id(family, rank, seed, wi, ai):
    return f"{family}_k{rank:03d}_s{seed}_w{wi}_a{ai}"


def fit_bank(x, y, weights, exact_backend="cuda"):
    started = time.perf_counter()
    prep = fit_normalizer(x, y)
    normalized = coordinates(prep, x)
    yn = (y.astype(np.float64) - prep["y_mean"]) / prep["y_std"]
    base_width = median_width(normalized)
    preparation_seconds = time.perf_counter() - started
    models, records, diagnostics = {}, {}, {}
    operation_costs = []

    def register(family, rank, seed, wi, ai, ids, beta, width, info=None):
        key = model_id(family, rank, seed, wi, ai)
        models[key] = {**prep, "centers": normalized[ids].astype(np.float32),
                       "coefficient": beta.astype(np.float32), "width": np.asarray(width, np.float32)}
        records[key] = {"family": family, "rank": rank, "actual_centers": len(ids), "seed": seed,
                        "width_factor": WIDTHS[wi], "width_index": wi, "alpha": ALPHAS[ai], "alpha_index": ai,
                        "fit_center_indices": ids.tolist(), "info": info or {}}
        return key

    for wi, factor in enumerate(WIDTHS):
        width = float(np.float32(base_width * factor))
        before = time.perf_counter()
        kernel = gaussian_kernel(normalized, normalized, width)
        teachers = exact_coefficients(kernel, yn, weights, ALPHAS, exact_backend).astype(np.float32)
        for ai, beta in enumerate(teachers):
            register("exact", 0, 17, wi, ai, np.arange(len(x)), beta, width)
        operation_costs.append({"operation": "kernel_and_exact_teacher_bank", "width_index": wi, "seconds": time.perf_counter() - before})
        for mode in ("random", "pivot", "rpchol"):
            seed_list = (17,) if mode == "pivot" else SEEDS
            for seed in seed_list:
                before = time.perf_counter()
                indices, _, residual = select_landmarks(kernel, weights, mode, RANKS[-1], seed)
                operation_costs.append({"operation": "landmark_selection", "mode": mode, "seed": seed, "width_index": wi,
                                        "seconds": time.perf_counter() - before, "selected": len(indices), "remaining_weighted_trace": float(residual.sum())})
                for nominal_rank in RANKS:
                    ids = indices[:nominal_rank]
                    before = time.perf_counter()
                    coefficients, info = nystrom_coefficients(kernel, ids, yn, weights, ALPHAS, backend="cpu")
                    for ai, beta in enumerate(coefficients):
                        register(f"nys_{mode}", nominal_rank, seed, wi, ai, ids, beta, width, info)
                    if mode == "rpchol":
                        projected, info = projection_coefficients(kernel, ids, teachers.astype(np.float64), backend="cpu")
                        for ai, beta in enumerate(projected):
                            beta = beta.astype(np.float32)
                            key = register("project_rpchol", nominal_rank, seed, wi, ai, ids, beta, width, info)
                            q, at_centers = residual_diagnostic(kernel, ids, teachers[ai], beta)
                            diagnostics[key] = {"norm_sq": q, "residual_values": at_centers}
                    operation_costs.append({"operation": "compact_readout_bank", "mode": mode, "seed": seed,
                                            "nominal_rank": nominal_rank, "width_index": wi, "seconds": time.perf_counter() - before})
    receipt = {"n_fit_rows": len(x), "base_width": base_width, "readout_configurations": len(models),
               "exact_backend": exact_backend, "compact_backend": "cpu", "preparation_seconds": preparation_seconds,
               "fit_bank_seconds": time.perf_counter() - started, "operation_costs": operation_costs, "models": records}
    return models, diagnostics, receipt


def flatten_bank(models, diagnostics):
    arrays = {}
    for key, model in models.items():
        for field, value in model.items():
            arrays[f"{key}__{field}"] = value
    for key, values in diagnostics.items():
        for field, value in values.items():
            arrays[f"{key}__{field}"] = value
    return arrays


def get_model(arrays, key):
    return {field: arrays[f"{key}__{field}"] for field in MODEL_FIELDS}


def evaluate_bank(models, diagnostics, receipt, x_query, row_indices):
    """No query labels: one query-kernel per width is reused over every readout."""
    arrays = {"row_indices": row_indices}
    maximum_bound_violation = -np.inf
    for wi in range(len(WIDTHS)):
        reference = models[model_id("exact", 0, 17, wi, 0)]
        query = coordinates(reference, x_query)
        kernel = gaussian_kernel(query, reference["centers"], float(reference["width"]))
        teacher_predictions = {ai: (kernel @ models[model_id("exact", 0, 17, wi, ai)]["coefficient"].astype(np.float64)) * reference["y_std"] + reference["y_mean"] for ai in range(len(ALPHAS))}
        for key, config in receipt["models"].items():
            if config["width_index"] != wi:
                continue
            model = models[key]
            feature = kernel[:, config["fit_center_indices"]]
            prediction = (feature @ model["coefficient"].astype(np.float64)) * model["y_std"] + model["y_mean"]
            arrays[f"pred__{key}"] = prediction
            if key in diagnostics:
                d = diagnostics[key]
                bounds = np.linalg.norm(residual_bound(feature, d["norm_sq"], d["residual_values"]) * model["y_std"], axis=1)
                difference = np.linalg.norm(prediction - teacher_predictions[config["alpha_index"]], axis=1)
                violation = float(np.max(difference - bounds))
                if violation > 1e-6:
                    raise ValueError(f"analytic teacher-bound check failed: {key}, violation {violation}")
                maximum_bound_violation = max(maximum_bound_violation, violation)
                arrays[f"bound__{key}"] = bounds
    return arrays, maximum_bound_violation


def make_adaptive(arrays, wi, ai, seed, tolerance):
    last = get_model(arrays, model_id("project_rpchol", RANKS[-1], seed, wi, ai))
    levels, previous = [], 0
    for nominal in RANKS:
        key = model_id("project_rpchol", nominal, seed, wi, ai)
        model = get_model(arrays, key)
        actual = len(model["centers"])
        if actual == previous:
            continue
        np.testing.assert_array_equal(model["centers"], last["centers"][:actual])
        levels.append((actual, model["coefficient"], arrays[f"{key}__norm_sq"], arrays[f"{key}__residual_values"]))
        previous = actual
    prep = {key: last[key] for key in ("x_mean", "x_std", "y_mean", "y_std", "width")}
    return pack_adaptive(prep, last["centers"], levels, tolerance)


def choose_from_trace(predictions, bounds, ranks, tolerance):
    count = predictions.shape[1]
    level = np.full(len(predictions), count - 1, np.int64)
    if tolerance > 0:
        for i in range(count - 2, -1, -1):
            level[bounds[:, i] <= tolerance] = i
    rows = np.arange(len(predictions))
    used = ranks[rows, level] if ranks.ndim == 2 else ranks[level]
    selected_bounds = bounds[rows, level]
    return predictions[rows, level], used, selected_bounds, (selected_bounds <= tolerance) & (tolerance > 0)
