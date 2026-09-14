"""Shared exact Nyström basis with a small input-conditioned residual readout."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_camera_support import row_weights, standardize
from chromaseed_kernel import coordinates, gaussian_kernel, predict_kernel
from chromaseed_perceptual import coupled_ridge, from_theta, geometry, make_basis, prepare
from scipy.linalg import cho_factor, cho_solve
from skin_local_search_train import weights_for

BASES = ("norm", "perceptual")
ROUTES = ("base", "uniform", "soft", "hard")
FAMILIES = tuple(f"{base}_{route}" for base in BASES for route in ROUTES)
SEEDS = (17, 29, 43)
LAMBDAS = (0.1, 1.0, 10.0)
RHOS = (0.25, 0.5, 1.0)
RANK = 128


def candidates(family):
    if family not in FAMILIES:
        raise ValueError("unknown family")
    zero = [dict(residual_lambda=10.0, rho=0.0)]
    return (
        zero
        if family.endswith("_base")
        else zero + [dict(residual_lambda=penalty, rho=r) for penalty in LAMBDAS for r in RHOS]
    )


def model_id(family, seed, residual_lambda=10.0, rho=0.0):
    base = family.split("_")[0]
    if rho == 0.0 or family.endswith("_base"):
        return f"{base}_base_s{seed}"
    return f"{family}_s{seed}_l{LAMBDAS.index(residual_lambda)}_r{RHOS.index(rho)}"


def gate_value(z, beta, route):
    score = z @ beta[1:].astype(np.float64) + float(beta[0])
    if route == "soft":
        return np.clip(score, -1.0, 1.0)
    if route == "hard":
        return np.where(score >= 0.0, 1.0, -1.0)
    raise ValueError("gate route required")


def fit_gate(z, person, site, camera):
    if len(np.unique(camera)) < 2:
        return None
    w = row_weights(person, site, camera)
    standardized, _, prep = standardize(z, z, w)
    a = np.column_stack([np.ones(len(z)), standardized])
    penalty = np.diag(np.r_[0.0, np.full(z.shape[1], 0.1)])
    h, rhs = a.T @ (w[:, None] * a) + penalty, a.T @ (w * np.where(camera == "SLR", 1.0, -1.0))
    theta = cho_solve(cho_factor(h, lower=True), rhs)
    coefficient = theta[1:] / prep["std"]
    return np.r_[theta[0] - prep["mean"] @ coefficient, coefficient].astype(np.float32)


def residual_solutions(design, residual, metric, weights, penalties):
    design, residual, metric, weights = [
        np.asarray(v, np.float64) for v in (design, residual, metric, weights)
    ]
    if (
        design.ndim != 2
        or residual.shape != (len(design), 3)
        or metric.shape != (3, 3)
        or weights.shape != (len(design),)
        or any(not np.isfinite(p) or p <= 0 for p in penalties)
        or np.any(weights <= 0)
        or not all(np.isfinite(v).all() for v in (design, residual, metric, weights))
    ):
        raise ValueError(
            "finite compatible residual system and positive penalties/weights required"
        )
    gram = design.T @ (weights[:, None] * design)
    gram = 0.5 * (gram + gram.T)
    cross = design.T @ (weights[:, None] * residual)
    eigen, q = np.linalg.eigh(gram)
    mu, v = np.linalg.eigh(metric)
    if mu.min() <= 0:
        raise ValueError("positive definite color metric required")
    clipped = int(np.sum(eigen < 0.0))
    eigen = np.maximum(eigen, 0.0)
    rotated = q.T @ cross @ v
    baseline = float(np.sum(weights * np.einsum("ni,ij,nj->n", residual, metric, residual)))
    solutions, records = {}, []
    for alpha in penalties:
        theta = q @ (rotated / (eigen[:, None] + alpha / mu[None, :])) @ v.T
        error = design @ theta - residual
        objective = float(
            np.sum(weights * np.einsum("ni,ij,nj->n", error, metric, error))
            + alpha * np.sum(theta**2)
        )
        normal = float(
            np.linalg.norm(gram @ theta @ metric + alpha * theta - cross @ metric)
            / max(np.linalg.norm(cross @ metric), 1.0)
        )
        solutions[alpha] = theta
        records.append(
            dict(
                residual_lambda=alpha,
                objective=objective,
                objective_minus_zero=objective - baseline,
                normal_relative_residual=normal,
            )
        )
    return solutions, dict(
        solutions=records, gram_eigenvalues_clipped=clipped, zero_objective=baseline
    )


def base_model(state, basis, base):
    if base == "norm":
        return basis["baselines"][0.1], np.eye(3)
    metric = np.average(geometry(state), axis=0, weights=state["weights"])
    theta, _ = coupled_ridge(
        basis["z"],
        state["yn"],
        np.broadcast_to(metric, (len(state["y"]), 3, 3)),
        state["weights"],
        0.1,
    )
    return from_theta(state, basis, theta), metric


def package_model(base, correction, beta, route, rho):
    if rho == 0.0 or route == "base" or correction is None:
        return dict(base)
    if route == "uniform":
        return {
            **base,
            "coefficient": (
                base["coefficient"].astype(np.float64) + rho * correction.astype(np.float64)
            ).astype(np.float32),
        }
    if route not in ("soft", "hard") or beta is None:
        raise ValueError("active route needs gate")
    return {
        **base,
        "correction": np.asarray(correction, np.float32),
        "gate_beta": np.asarray(beta, np.float32),
        "gate_mode": np.asarray(1 if route == "soft" else 2, np.uint8),
        "rho": np.asarray(rho, np.float32),
    }


def predict(model, x):
    if "correction" not in model:
        return predict_kernel(model, x)
    z = coordinates(model, x)
    k = gaussian_kernel(z, model["centers"], float(model["width"]))
    value = k @ model["coefficient"].astype(np.float64)
    route = "soft" if int(model["gate_mode"]) == 1 else "hard"
    gate = gate_value(z, model["gate_beta"], route)
    value += float(model["rho"]) * gate[:, None] * (k @ model["correction"].astype(np.float64))
    return value * model["y_std"] + model["y_mean"]


def correction_bank(state, basis, model, metric, beta, route, penalties):
    signal = (
        np.ones(len(state["y"]))
        if route == "uniform"
        else gate_value(state["normalized"], beta, route)
    )
    design = signal[:, None] * basis["z"]
    k = gaussian_kernel(state["normalized"], model["centers"], float(model["width"]))
    residual = state["yn"] - k @ model["coefficient"].astype(np.float64)
    theta, info = residual_solutions(design, residual, metric, state["weights"], penalties)
    return {a: (basis["whitening"] @ t).astype(np.float32) for a, t in theta.items()}, info


def fit_single(x, y, person, site, camera, family, seed, residual_lambda, rho, rank=RANK):
    if (
        family not in FAMILIES
        or not np.isfinite(residual_lambda)
        or residual_lambda <= 0
        or rho not in (0.0, *RHOS)
    ):
        raise ValueError("registered family, strength and positive residual penalty required")
    started = time.perf_counter()
    weights = weights_for(person, site)
    state = prepare(x, y, weights)
    basis = make_basis(state, seed, 1, rank, (0.1,))
    base, route = family.split("_")
    model, metric = base_model(state, basis, base)
    info = dict(residual_solutions=0, gate_fits=0, fallback="none")
    if rho == 0.0 or route == "base":
        info["fallback"] = "zero"
    elif route in ("soft", "hard") and len(np.unique(camera)) < 2:
        info["fallback"] = "single_camera"
    else:
        beta = (
            fit_gate(state["normalized"], person, site, camera)
            if route in ("soft", "hard")
            else None
        )
        correction, details = correction_bank(
            state, basis, model, metric, beta, route, (residual_lambda,)
        )
        model = package_model(model, correction[residual_lambda], beta, route, rho)
        info.update(residual_solutions=1, gate_fits=int(beta is not None), residual=details)
    return model, {**info, "fit_seconds": time.perf_counter() - started}


def fit_bank(x, y, person, site, camera):
    started = time.perf_counter()
    state = prepare(x, y, weights_for(person, site))
    beta = fit_gate(state["normalized"], person, site, camera)
    models, records, operations = {}, {}, []
    for seed in SEEDS:
        basis = make_basis(state, seed, 1, RANK, (0.1,))
        for base in BASES:
            model, metric = base_model(state, basis, base)
            name = model_id(f"{base}_base", seed)
            models[name] = model
            records[name] = dict(
                family=f"{base}_base", seed=seed, residual_lambda=10.0, rho=0.0, fallback="base"
            )
            for route in ("uniform", "soft", "hard"):
                corrections, details = None, None
                if route == "uniform" or beta is not None:
                    corrections, details = correction_bank(
                        state, basis, model, metric, beta, route, LAMBDAS
                    )
                    operations.append(dict(base=base, route=route, seed=seed, **details))
                for alpha in LAMBDAS:
                    for rho in RHOS:
                        family = f"{base}_{route}"
                        name = model_id(family, seed, alpha, rho)
                        models[name] = package_model(
                            model, corrections[alpha] if corrections else None, beta, route, rho
                        )
                        records[name] = dict(
                            family=family,
                            seed=seed,
                            residual_lambda=alpha,
                            rho=rho,
                            fallback="single_camera" if corrections is None else "none",
                        )
    assert len(models) == 168
    return models, dict(
        models=records,
        operations=operations,
        n_fit_rows=len(x),
        gate_fits=int(beta is not None),
        residual_solutions=sum(len(o["solutions"]) for o in operations),
        perceptual_base_solves=3,
        fit_bank_seconds=time.perf_counter() - started,
    )


def flatten(models):
    return {
        f"{name}__{field}": value
        for name, model in models.items()
        for field, value in model.items()
    }


def unpack(arrays, name):
    prefix = name + "__"
    return {key[len(prefix) :]: value for key, value in arrays.items() if key.startswith(prefix)}
