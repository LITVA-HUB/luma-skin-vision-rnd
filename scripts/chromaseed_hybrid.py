"""Staged analytic corrections on raw/shared-projected landmark geometry."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_affine import color_metrics, design_matrix, export_model, gram_stats, solve_stats
from chromaseed_gated import fit_gate
from chromaseed_gated import predict as raw_predict
from chromaseed_kernel import coordinates, gaussian_kernel
from chromaseed_perceptual import make_basis, prepare
from chromaseed_projection import exact_width, export, project, projection
from chromaseed_projection import predict as old_predict
from skin_local_search_train import weights_for

LOSSES, SEEDS, ALPHAS = ("norm", "perceptual"), (17, 29, 43), (0.1, 1.0, 10.0)
RHOS, POWERS = (0.25, 0.5, 1.0), (1, 4)
KINDS = ("raw", "projected", "blend", "uniform", "support")
MODES = {"blend": 1, "uniform": 2, "support": 3}


def support(kernel, power):
    kernel = np.asarray(kernel, np.float64)
    if power not in POWERS or kernel.ndim != 2 or kernel.shape[1] < 1:
        raise ValueError("kernel matrix and registered support power required")
    if not np.isfinite(kernel).all() or np.any(kernel < 0) or np.any(kernel > 1 + 1e-12):
        raise ValueError("finite unit-interval similarities required")
    return np.mean(kernel, axis=1) ** power


def shared_basis(q, ids, width):
    q, ids = np.asarray(q, np.float64), np.asarray(ids, np.int64)
    if q.ndim != 2 or q.shape[1] != 16 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("latent16 and unique original row indices required")
    if np.any(ids < 0) or np.any(ids >= len(q)):
        raise ValueError("center index outside original fit")
    columns = gaussian_kernel(q, q[ids], width)
    columns[ids, np.arange(len(ids))] = 1.0
    km = columns[ids]
    eigen, vectors = np.linalg.eigh(0.5 * (km + km.T))
    keep = eigen > 1e-8 * max(float(eigen.max()), np.finfo(float).tiny)
    white = vectors[:, keep] / np.sqrt(eigen[keep])
    return dict(
        ids=ids.copy(),
        centers=q[ids].astype(np.float32),
        width=width,
        whitening=white,
        z=columns @ white,
    )


def model_id(loss, seed, kind, alpha=0.1, rho=0.0, power=0):
    if kind == "raw" or (kind in MODES and rho == 0):
        return f"{loss}_raw_s{seed}"
    name = f"{loss}_{kind}_s{seed}_a{ALPHAS.index(alpha)}"
    return name if kind == "projected" else f"{name}_r{RHOS.index(rho)}_p{power}"


def settings(kind):
    if kind not in KINDS:
        raise ValueError("registered family required")
    raw = dict(kind="raw", alpha=0.1, rho=0.0, power=0)
    if kind == "raw":
        return [raw]
    if kind == "projected":
        return [dict(kind=kind, alpha=a, rho=1.0, power=0) for a in ALPHAS]
    return [raw] + [
        dict(kind=kind, alpha=a, rho=r, power=p)
        for a in ALPHAS
        for r in RHOS
        for p in (POWERS if kind == "support" else (0,))
    ]


def choose(candidates):
    if not candidates:
        raise ValueError("nonempty registered candidates required")
    best = min(c["clean"] for c in candidates)
    return min(
        (c for c in candidates if c["clean"] <= best + 1e-5),
        key=lambda c: (c["numeric_bytes"], c["clean"], c["p90"], -c["alpha"], c["rho"], c["power"]),
    )


def package(raw, branch, kind, rho, power):
    if (
        kind not in MODES
        or rho not in (0.0, *RHOS)
        or power not in ((0,) if kind != "support" else POWERS)
    ):
        raise ValueError("registered correction setting required")
    if rho == 0:
        return dict(raw)
    if kind == "blend" and rho == 1:
        return dict(branch)
    projected_centers = (
        (raw["centers"].astype(np.float64) - branch["projection_mean"].astype(np.float64))
        @ branch["projection"].astype(np.float64)
    ).astype(np.float32)
    np.testing.assert_array_equal(projected_centers, branch["centers"])
    for k in ("x_mean", "x_std", "y_mean", "y_std"):
        np.testing.assert_array_equal(raw[k], branch[k])
    if ("gate_beta" in raw) != ("gate_beta" in branch):
        raise ValueError("matched original joint structure required")
    if "gate_beta" in raw:
        np.testing.assert_array_equal(raw["gate_beta"], branch["gate_beta"])
    extra = dict(
        latent_projection=branch["projection"],
        latent_mean=branch["projection_mean"],
        latent_width=branch["width"],
        latent_coefficient=branch["coefficient"],
        hybrid_mode=np.array(MODES[kind], np.uint8),
        mix=np.array(rho, np.float32),
        support_power=np.array(power, np.uint8),
    )
    if "correction" in branch:
        extra["latent_correction"] = branch["correction"]
    return dict(**raw, **extra)


def predict(model, x):
    if "hybrid_mode" not in model:
        return old_predict(model, x)
    z = coordinates(model, x)
    raw_k = gaussian_kernel(z, model["centers"], float(model["width"]))
    p, mean = model["latent_projection"].astype(np.float64), model["latent_mean"].astype(np.float64)
    q = ((z - mean) @ p).astype(np.float32).astype(np.float64)
    centers = ((model["centers"].astype(np.float64) - mean) @ p).astype(np.float32)
    k = gaussian_kernel(q, centers, float(model["latent_width"]))
    base, correction = (
        raw_k @ model["coefficient"].astype(np.float64),
        k @ model["latent_coefficient"].astype(np.float64),
    )
    if "gate_beta" in model:
        signal = np.clip(
            z @ model["gate_beta"][1:].astype(np.float64) + float(model["gate_beta"][0]), -1, 1
        )[:, None]
        base += signal * (raw_k @ model["correction"].astype(np.float64))
        correction += signal * (k @ model["latent_correction"].astype(np.float64))
    rho = float(model["mix"])
    mode = int(model["hybrid_mode"])
    if mode == 1:
        value = (1 - rho) * base + rho * correction
    elif mode in (2, 3):
        scale = support(raw_k, int(model["support_power"]))[:, None] if mode == 3 else 1.0
        value = base + rho * scale * correction
    else:
        raise ValueError("unsupported hybrid mode")
    return value * model["y_std"] + model["y_mean"]


def raw_fit(state, beta, metrics, seed, rank):
    b = make_basis(state, seed, 1, rank, (0.1,))
    score = (
        None if beta is None else state["normalized"] @ beta[1:].astype(np.float64) + float(beta[0])
    )
    solved, info = solve_stats(
        gram_stats(design_matrix(b["z"], score, beta is not None), state["yn"], state["weights"]),
        metrics,
        (0.1,),
    )
    models = {
        loss: export_model(state, b, theta, beta, beta is not None)
        for (loss, _), theta in solved.items()
    }
    return models, b, score, info


def projection_fit(state, x):
    payload, info = projection(state["normalized"], state["weights"], 16, 0.5)
    q = project({**state["prep"], **payload}, x)
    width, wi = exact_width(q)
    return payload, q, float(np.float32(width)), dict(**info, width_info=wi)


def fit_bank(x, y, person, site, camera, rank=128):
    start = time.perf_counter()
    state = prepare(x, y, weights_for(person, site))
    beta = fit_gate(state["normalized"], person, site, camera)
    metric = color_metrics(state)
    payload, q, width, pi = projection_fit(state, x)
    models, metadata, operations, bases = {}, {}, [], []

    def add(loss, seed, kind, alpha, rho, power, m):
        name = model_id(loss, seed, kind, alpha, rho, power)
        assert name not in models
        models[name] = m
        metadata[name] = dict(loss=loss, seed=seed, kind=kind, alpha=alpha, rho=rho, power=power)

    for seed in SEEDS:
        raw, b, signal, oi = raw_fit(state, beta, metric, seed, rank)
        operations.append(dict(seed=seed, kind="raw", **oi))
        pb = shared_basis(q, b["ids"], width)
        bases.append(
            dict(
                seed=seed,
                fit_center_indices=b["ids"].tolist(),
                actual_centers=len(b["ids"]),
                raw_rank=b["whitening"].shape[1],
                projected_rank=pb["whitening"].shape[1],
            )
        )
        design = design_matrix(pb["z"], signal, beta is not None)
        projected, oi = solve_stats(
            gram_stats(design, state["yn"], state["weights"]), metric, ALPHAS
        )
        operations.append(dict(seed=seed, kind="projected", **oi))
        for loss in LOSSES:
            add(loss, seed, "raw", 0.1, 0.0, 0, raw[loss])
            for alpha in ALPHAS:
                m = export(state, payload, pb, projected[(loss, alpha)], beta, beta is not None)
                add(loss, seed, "projected", alpha, 1.0, 0, m)
                for rho in RHOS:
                    add(loss, seed, "blend", alpha, rho, 0, package(raw[loss], m, "blend", rho, 0))
            residual = (
                state["yn"]
                - (raw_predict(raw[loss], x) - state["prep"]["y_mean"]) / state["prep"]["y_std"]
            )
            raw_k = gaussian_kernel(
                state["normalized"], raw[loss]["centers"], float(raw[loss]["width"])
            )
            for kind, power in (("uniform", 0), ("support", 1), ("support", 4)):
                d = design if power == 0 else support(raw_k, power)[:, None] * design
                solved, oi = solve_stats(
                    gram_stats(d, residual, state["weights"]), {loss: metric[loss]}, ALPHAS
                )
                operations.append(dict(seed=seed, kind=kind, loss=loss, power=power, **oi))
                for alpha in ALPHAS:
                    m = export(state, payload, pb, solved[(loss, alpha)], beta, beta is not None)
                    for rho in RHOS:
                        add(
                            loss,
                            seed,
                            kind,
                            alpha,
                            rho,
                            power,
                            package(raw[loss], m, kind, rho, power),
                        )
    assert len(models) == 240
    return models, dict(
        models=metadata,
        operations=operations,
        bases=bases,
        projection_info=pi,
        new_coefficient_solutions=sum(len(o["solutions"]) for o in operations),
        gram_decompositions=len(operations),
        basis_preparations=6,
        exact_width_calculations=2,
        covariance_eigendecompositions=1,
        gate_fits=int(beta is not None),
        auxiliary_baseline_solves=3,
        auxiliary_theta_solves=3,
        n_fit_rows=len(x),
        original_weight_mass=float(state["weights"].sum()),
        fit_bank_seconds=time.perf_counter() - start,
    )


def fit_single(x, y, person, site, camera, loss, seed, kind, alpha=0.1, rho=0.0, power=0, rank=128):
    if (
        loss not in LOSSES
        or seed not in SEEDS
        or dict(kind=kind, alpha=alpha, rho=rho, power=power) not in settings(kind)
    ):
        raise ValueError("registered single-fit setting required")
    start = time.perf_counter()
    state = prepare(x, y, weights_for(person, site))
    beta = fit_gate(state["normalized"], person, site, camera)
    metric = color_metrics(state, (loss,))
    raw, b, signal, _ = raw_fit(state, beta, metric, seed, rank)
    m = raw[loss]
    if kind != "raw":
        payload, q, width, _ = projection_fit(state, x)
        pb = shared_basis(q, b["ids"], width)
        design = design_matrix(pb["z"], signal, beta is not None)
        target = state["yn"]
        if kind in ("uniform", "support"):
            target = target - (raw_predict(m, x) - state["prep"]["y_mean"]) / state["prep"]["y_std"]
        if kind == "support":
            raw_k = gaussian_kernel(state["normalized"], m["centers"], float(m["width"]))
            design = support(raw_k, power)[:, None] * design
        sol, _ = solve_stats(gram_stats(design, target, state["weights"]), metric, (alpha,))
        branch = export(state, payload, pb, sol[(loss, alpha)], beta, beta is not None)
        m = branch if kind == "projected" else package(m, branch, kind, rho, power)
    return m, dict(fit_seconds=time.perf_counter() - start)
