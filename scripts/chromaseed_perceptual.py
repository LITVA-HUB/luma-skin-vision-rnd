"""Compact kernel readouts with local perceptual geometry and fit-only IRLS."""
from __future__ import annotations

import time

import numpy as np
from chromaseed_condensed_exact import condensed_width
from chromaseed_fast_kernel import ALPHAS, SEEDS, WIDTHS, _landmark_fit, payload, solve_columns
from chromaseed_kernel import EIGEN_FLOOR, coordinates, fit_normalizer
from scipy.linalg import cho_factor, cho_solve

from luma_skin_vision.color import delta_e00

FAMILIES = ("norm_mse", "constant_de2", "local_de2", "local_irls", "midpoint_irls")
CHECKPOINTS = (1, 4, 16)
RANK = 128
TAU = 2.


def local_tensor(anchors, h=.001):
    anchors = np.asarray(anchors, np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 3 or not np.isfinite(anchors).all() or h <= 0:
        raise ValueError("finite Lab anchors and positive finite-difference step required")
    eye = np.eye(3) * h
    offsets = [v for axis in eye for v in (axis, -axis)]
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        offsets.extend([eye[i] + eye[j], -eye[i] - eye[j], eye[i] - eye[j], -eye[i] + eye[j]])
    q = delta_e00(anchors[:, None] + np.asarray(offsets), anchors[:, None]) ** 2
    metric = np.zeros((len(anchors), 3, 3))
    for i in range(3):
        metric[:, i, i] = (q[:, 2 * i] + q[:, 2 * i + 1]) / (2 * h * h)
    for index, (i, j) in enumerate(pairs):
        a, b, c, d = q[:, 6 + 4 * index:10 + 4 * index].T
        metric[:, i, j] = metric[:, j, i] = (a + b - c - d) / (8 * h * h)
    eigen, vectors = np.linalg.eigh(metric)
    clipped = int(np.sum(eigen < 1e-8))
    info = {"clipped_eigenvalues": clipped, "minimum_raw_eigenvalue": float(eigen.min())}
    if clipped:
        metric = (vectors * np.maximum(eigen, 1e-8)[:, None]) @ vectors.transpose(0, 2, 1)
    return metric, info


def coupled_ridge(z, target, metric, weights, alpha):
    z, target, metric, weights = [np.asarray(v, np.float64) for v in (z, target, metric, weights)]
    if target.shape != (len(z), 3) or metric.shape != (len(z), 3, 3) or weights.shape != (len(z),):
        raise ValueError("incompatible coupled readout shapes")
    if alpha <= 0 or np.any(weights <= 0) or not all(np.isfinite(v).all() for v in (z, target, metric, weights)):
        raise ValueError("positive regularization/weights and finite arrays required")
    rank = z.shape[1]
    system = np.empty((3 * rank, 3 * rank))
    for a in range(3):
        for b in range(a, 3):
            block = z.T @ ((weights * metric[:, a, b])[:, None] * z)
            system[a::3, b::3] = block
            system[b::3, a::3] = block.T
    system.flat[::len(system) + 1] += alpha
    rhs = (z.T @ (weights[:, None] * np.einsum("nij,nj->ni", metric, target))).ravel()
    solution = cho_solve(cho_factor(system, lower=True, check_finite=False), rhs, check_finite=False)
    residual = float(np.linalg.norm(system @ solution - rhs) / max(np.linalg.norm(rhs), 1.))
    return solution.reshape(rank, 3), residual


def prepare(x, y, weights):
    prep = fit_normalizer(x, y)
    normalized = coordinates(prep, x)
    base, width_info = condensed_width(normalized)
    weights = np.asarray(weights, np.float64)
    if weights.shape != (len(x),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("finite positive fit weights required")
    return {"prep": prep, "normalized": normalized, "base": base, "width_info": width_info,
            "y": np.asarray(y, np.float64), "yn": (np.asarray(y, np.float64) - prep["y_mean"]) / prep["y_std"], "weights": weights}


def geometry(state):
    if "tensor" not in state:
        raw, info = local_tensor(state["y"])
        std = state["prep"]["y_std"].astype(np.float64)
        normalized = raw * std[None, :, None] * std[None, None, :]
        scale = float(np.average(np.trace(normalized, axis1=1, axis2=2), weights=state["weights"]) / 3)
        state.update(tensor=raw, tensor_info=info, metric=normalized / scale, metric_scale=scale)
    return state["metric"]


def make_basis(state, seed, width_index, rank=RANK, alphas=ALPHAS):
    width = float(np.float32(state["base"] * WIDTHS[width_index]))
    ids, columns, info = _landmark_fit(state["normalized"], state["weights"], "column_exact", width, rank, seed)
    sub = columns[ids]
    values, vectors = np.linalg.eigh(.5 * (sub + sub.T))
    keep = values > EIGEN_FLOOR * max(float(values.max()), np.finfo(float).tiny)
    whitening = vectors[:, keep] / np.sqrt(values[keep])
    z = columns @ whitening
    gram = z.T @ (state["weights"][:, None] * z)
    cross = z.T @ (state["weights"][:, None] * state["yn"])
    coefficients, solve_info = solve_columns(columns, ids, state["yn"], state["weights"], alphas)
    baselines, theta = {}, {}
    for alpha, coefficient in zip(alphas, coefficients, strict=True):
        baselines[alpha] = payload(state["prep"], state["normalized"], ids, coefficient, width)
        theta[alpha] = np.linalg.solve(gram + alpha * np.eye(len(gram)), cross)
    return {"ids": ids, "width": width, "whitening": whitening, "z": z, "baselines": baselines, "theta": theta,
            "info": {**info, **solve_info}}


def from_theta(state, basis, theta):
    return payload(state["prep"], state["normalized"], basis["ids"], basis["whitening"] @ theta, basis["width"])


def objective(state, basis, theta, alpha, family):
    error = (basis["z"] @ theta - state["yn"]) * state["prep"]["y_std"]
    if family == "local_irls":
        squared = np.einsum("ni,nij,nj->n", error, state["tensor"], error)
    elif family == "midpoint_irls":
        squared = delta_e00(state["y"] + error, state["y"]) ** 2
    else:
        raise ValueError("iterative family required")
    loss = 2 * TAU * (np.sqrt(np.maximum(squared, 0.) + TAU ** 2) - TAU)
    return float(state["weights"] @ loss / state["metric_scale"] + alpha * np.sum(theta ** 2))


def trajectory(state, basis, family, alpha, checkpoints=CHECKPOINTS):
    fixed = geometry(state)
    theta = basis["theta"][alpha].copy()
    old_objective = objective(state, basis, theta, alpha, family)
    history = [{"step": 0, "objective": old_objective, "accepted_length": 0., "executed": False}]
    snapshots, stopped, solves = {}, False, 0
    std = state["prep"]["y_std"].astype(np.float64)
    clips = state["tensor_info"]["clipped_eigenvalues"]
    for step in range(1, max(checkpoints) + 1):
        length, residual = 0., 0.
        executed = not stopped
        if executed:
            error = (basis["z"] @ theta - state["yn"]) * std
            if family == "midpoint_irls":
                raw, tensor_info = local_tensor(state["y"] + .5 * error)
                clips += tensor_info["clipped_eigenvalues"]
                metric = raw * std[None, :, None] * std[None, None, :] / state["metric_scale"]
            else:
                raw, metric = state["tensor"], fixed
            squared = np.einsum("ni,nij,nj->n", error, raw, error)
            robust = TAU / np.sqrt(np.maximum(squared, 0.) + TAU ** 2)
            proposal, residual = coupled_ridge(basis["z"], state["yn"], metric, state["weights"] * robust, alpha)
            solves += 1
            for candidate_length in (1., .5, .25, .125, .0625, .03125, .015625):
                candidate = theta + candidate_length * (proposal - theta)
                value = objective(state, basis, candidate, alpha, family)
                if np.isfinite(value) and value < old_objective:
                    theta, old_objective, length = candidate, value, candidate_length
                    break
            if length == 0:
                stopped = True
        history.append({"step": step, "objective": old_objective, "accepted_length": length,
                        "normal_equation_relative_residual": residual, "executed": executed})
        if step in checkpoints:
            snapshots[step] = theta.copy()
    return snapshots, {"trajectory": history, "executed_solves": solves, "stopped": stopped, "tensor_eigenvalue_clips": clips}


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


def key(family, seed, wi, ai, steps):
    if family.endswith("irls") and steps == 0:
        family = "norm_mse"
    return f"{family}_s{seed}_w{wi}_a{ai}_t{steps}"


def fit_bank(x, y, weights):
    started = time.perf_counter()
    state = prepare(x, y, weights)
    models, records, operations = {}, {}, []
    for seed in SEEDS:
        for wi in range(len(WIDTHS)):
            before = time.perf_counter()
            basis = make_basis(state, seed, wi)
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
                    snapshots, details = trajectory(state, basis, family, alpha)
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
