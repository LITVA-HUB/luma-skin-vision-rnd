"""Expanded positive ridge grid; numerical operations reuse frozen P helpers."""
from __future__ import annotations

import time

import numpy as np
from chromaseed_perceptual import (
    CHECKPOINTS,
    FAMILIES,
    RANK,
    SEEDS,
    WIDTHS,
    coupled_ridge,
    from_theta,
    geometry,
    key,
    make_basis,
    prepare,
    trajectory,
)

ALPHAS = (.0001, .0003, .001, .003, .01, .03, .1, 1., 10.)


def fit_single(x, y, weights, family, seed, width_index, alpha_index, steps, rank=RANK):
    if family not in FAMILIES or steps < 0:
        raise ValueError("registered family and nonnegative steps required")
    started = time.perf_counter()
    alpha = ALPHAS[alpha_index]
    state = prepare(x, y, weights)
    basis = make_basis(state, seed, width_index, rank, (alpha,))
    info = {"executed_solves": 0, "basis": basis["info"]}
    if family == "norm_mse" or (family.endswith("irls") and steps == 0):
        model = basis["baselines"][alpha]
    elif family in ("constant_de2", "local_de2"):
        metric = geometry(state)
        if family == "constant_de2":
            metric = np.broadcast_to(np.average(metric, axis=0, weights=state["weights"]), metric.shape)
        theta, residual = coupled_ridge(basis["z"], state["yn"], metric, state["weights"], alpha)
        model = from_theta(state, basis, theta)
        info.update(executed_solves=1, normal_equation_relative_residual=residual)
    else:
        snapshots, details = trajectory(state, basis, family, alpha, (steps,))
        model = from_theta(state, basis, snapshots[steps])
        info.update(details)
    return model, {**info, "fit_seconds": time.perf_counter() - started}


def fit_bank(x, y, weights):
    started = time.perf_counter()
    state = prepare(x, y, weights)
    models, records, operations = {}, {}, []
    for seed in SEEDS:
        for wi in range(len(WIDTHS)):
            before = time.perf_counter()
            basis = make_basis(state, seed, wi, alphas=ALPHAS)
            operations.append({"kind": "basis", "seed": seed, "width_index": wi, "seconds": time.perf_counter() - before, **basis["info"]})
            for ai, alpha in enumerate(ALPHAS):
                name = key("norm_mse", seed, wi, ai, 0)
                models[name] = basis["baselines"][alpha]
                records[name] = {"family": "norm_mse", "seed": seed, "width_index": wi, "alpha_index": ai, "steps": 0}
                metric = geometry(state)
                for family in ("constant_de2", "local_de2"):
                    before = time.perf_counter()
                    g = metric if family == "local_de2" else np.broadcast_to(np.average(metric, axis=0, weights=weights), metric.shape)
                    theta, residual = coupled_ridge(basis["z"], state["yn"], g, weights, alpha)
                    name = key(family, seed, wi, ai, 1)
                    models[name] = from_theta(state, basis, theta)
                    records[name] = {"family": family, "seed": seed, "width_index": wi, "alpha_index": ai, "steps": 1}
                    operations.append({"kind": "quadratic", "key": name, "seconds": time.perf_counter() - before,
                                       "executed_solves": 1, "normal_equation_relative_residual": residual})
                for family in ("local_irls", "midpoint_irls"):
                    before = time.perf_counter()
                    snapshots, details = trajectory(state, basis, family, alpha, checkpoints=CHECKPOINTS)
                    for step, theta in snapshots.items():
                        name = key(family, seed, wi, ai, step)
                        models[name] = from_theta(state, basis, theta)
                        records[name] = {"family": family, "seed": seed, "width_index": wi, "alpha_index": ai, "steps": step}
                    operations.append({"kind": "iterative", "family": family, "seed": seed, "width_index": wi, "alpha_index": ai,
                                       "seconds": time.perf_counter() - before, **details})
            print(f"  BASIS seed{seed} width{wi} done", flush=True)
    return models, {"fit_bank_seconds": time.perf_counter() - started, "models": records, "operations": operations,
                    "n_fit_rows": len(x), "readout_configurations": len(models), "tensor_info": state["tensor_info"],
                    "metric_scale": state["metric_scale"], "base_width": state["base"], "width_info": state["width_info"]}

