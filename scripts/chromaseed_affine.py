"""Compact analytical static/joint readouts with fit-only mild affine augmentation."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_gate_stability import ANCHORS, affine_features
from chromaseed_gated import fit_gate, package_model
from chromaseed_gated import predict as gated_predict
from chromaseed_kernel import coordinates, gaussian_kernel
from chromaseed_perceptual import from_theta, geometry, make_basis, prepare
from skin_local_search_train import weights_for

BASES = ("norm", "perceptual")
FAMILIES = tuple(f"{b}_{s}" for b in BASES for s in ("static", "joint_soft"))
ALPHAS = (0.1, 1.0, 10.0)
ETAS = (0.0, 0.25, 0.5, 0.75)
SEEDS = (17, 29, 43)
TRAIN_DOSES = (1 / 255, 4 / 255)
POLICIES = ("clean", "guarded")
G_CONTROLS = ("norm_base", "norm_soft", "perceptual_base", "perceptual_soft")


def weighted_copies(w, eta):
    w = np.asarray(w, np.float64)
    if (
        w.ndim != 1
        or not np.isfinite(w).all()
        or np.any(w <= 0)
        or not np.isfinite(eta)
        or not 0 <= eta < 1
    ):
        raise ValueError("positive finite original weights and0<=eta<1 required")
    return np.vstack([(1 - eta) * w, np.broadcast_to(eta * w / 16, (16, len(w)))])


def gram_stats(z, y, w):
    z, y, w = (np.asarray(v, np.float64) for v in (z, y, w))
    if (
        z.ndim != 2
        or y.shape != (len(z), 3)
        or w.shape != (len(z),)
        or np.any(w < 0)
        or w.sum() <= 0
        or not all(np.isfinite(v).all() for v in (z, y, w))
    ):
        raise ValueError("compatible finite design/targets and nonnegative mass required")
    return z.T @ (w[:, None] * z), z.T @ (w[:, None] * y), y.T @ (w[:, None] * y)


def combine(original, augmented, eta):
    if not np.isfinite(eta) or not 0 <= eta < 1:
        raise ValueError("0<=eta<1 required")
    if eta == 0:
        return tuple(v.copy() for v in original)
    return tuple((1 - eta) * a + eta * b for a, b in zip(original, augmented, strict=True))


def solve_stats(stats, metrics, penalties):
    h, c, yy = (np.asarray(v, np.float64) for v in stats)
    if (
        h.ndim != 2
        or h.shape[0] != h.shape[1]
        or c.shape != (len(h), 3)
        or yy.shape != (3, 3)
        or not all(np.isfinite(v).all() for v in (h, c, yy))
    ):
        raise ValueError("finite compatible sufficient statistics required")
    if any(not np.isfinite(a) or a <= 0 for a in penalties):
        raise ValueError("positive finite ridge required")
    h = 0.5 * (h + h.T)
    eigen, q = np.linalg.eigh(h)
    if eigen.min() < -1e-8 * max(float(eigen.max()), 1):
        raise ValueError("feature Gram is not positive semidefinite")
    clipped = int((eigen < 0).sum())
    eigen = np.maximum(eigen, 0)
    solutions, records = {}, []
    for name, metric in metrics.items():
        metric = np.asarray(metric, np.float64)
        if (
            metric.shape != (3, 3)
            or not np.isfinite(metric).all()
            or not np.allclose(metric, metric.T, atol=1e-12, rtol=0)
        ):
            raise ValueError("symmetric finite color metric required")
        mu, v = np.linalg.eigh(metric)
        if mu.min() <= 0:
            raise ValueError("positive color metric required")
        rotated = q.T @ c @ v
        zero = float(np.sum(yy * metric.T))
        for alpha in penalties:
            theta = q @ (rotated / (eigen[:, None] + alpha / mu[None])) @ v.T
            objective = float(
                np.sum((theta.T @ h @ theta) * metric.T)
                - 2 * np.sum(theta * (c @ metric))
                + zero
                + alpha * np.sum(theta**2)
            )
            normal = float(
                np.linalg.norm(h @ theta @ metric + alpha * theta - c @ metric)
                / max(np.linalg.norm(c @ metric), 1)
            )
            solutions[(name, alpha)] = theta
            records.append(
                dict(
                    loss=name,
                    alpha=alpha,
                    objective=objective,
                    objective_minus_zero=objective - zero,
                    normal_residual=normal,
                )
            )
    return solutions, dict(gram_eigenvalues_clipped=clipped, solutions=records)


def design_matrix(z, score, joint):
    if not joint or score is None:
        return z
    return np.column_stack([z, np.clip(score, -1.0, 1.0)[:, None] * z])


def export_model(state, basis, theta, beta, joint):
    dim = basis["whitening"].shape[1]
    active = joint and beta is not None
    if theta.shape != ((2 if active else 1) * dim, 3):
        raise ValueError("coefficient dimension does not match active structure")
    model = from_theta(state, basis, theta[:dim])
    if active:
        correction = basis["whitening"] @ theta[dim:]
        model = package_model(model, correction, beta, "soft", 1.0)
    return model


def predict(model, x):
    if "constant_lab" in model:
        x = np.asarray(x, np.float32)
        if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
            raise ValueError("finite color36 batch required")
        return np.broadcast_to(model["constant_lab"].astype(np.float64), (len(x), 3)).copy()
    return gated_predict(model, x)


def model_id(family, seed, alpha, eta):
    return f"{family}_s{seed}_a{ALPHAS.index(alpha)}_e{ETAS.index(eta)}"


def choose(candidates, policy):
    if policy not in POLICIES or not candidates:
        raise ValueError("registered policy and nonempty candidates required")
    if policy == "clean":
        return min(candidates, key=lambda r: (r["clean"], r["eta"], -r["alpha"]))
    anchor = min(r["clean"] for r in candidates if r["eta"] == 0)
    eligible = [r for r in candidates if r["clean"] <= anchor + 0.05]
    return min(eligible, key=lambda r: (r["robust"], r["clean"], r["eta"], -r["alpha"]))


def sufficient_stats(state, basis, x, beta, structures, augment):
    score = (
        None if beta is None else state["normalized"] @ beta[1:].astype(np.float64) + float(beta[0])
    )
    original = {
        j: gram_stats(design_matrix(basis["z"], score, j), state["yn"], state["weights"])
        for j in structures
    }
    augmented = {j: tuple(np.zeros_like(a) for a in original[j]) for j in structures}
    if augment:
        centers = state["normalized"][basis["ids"]].astype(np.float32)
        for t in TRAIN_DOSES:
            for anchor in ANCHORS:
                xx = affine_features(x, t, anchor)
                z = coordinates(state["prep"], xx)
                columns = gaussian_kernel(z, centers, basis["width"])
                features = columns @ basis["whitening"]
                s = None if beta is None else z @ beta[1:].astype(np.float64) + float(beta[0])
                for joint in structures:
                    stats = gram_stats(
                        design_matrix(features, s, joint), state["yn"], state["weights"]
                    )
                    for total, value in zip(augmented[joint], stats, strict=True):
                        total += value / 16
    return original, augmented


def color_metrics(state, bases=BASES):
    metrics = {}
    if "norm" in bases:
        metrics["norm"] = np.eye(3)
    if "perceptual" in bases:
        metrics["perceptual"] = np.average(geometry(state), axis=0, weights=state["weights"])
    return metrics


def fit_bank(x, y, person, site, camera, rank=128):
    started = time.perf_counter()
    state = prepare(x, y, weights_for(person, site))
    beta = fit_gate(state["normalized"], person, site, camera)
    metrics = color_metrics(state)
    structures = (False, True) if beta is not None else (False,)
    models, metadata, operations = {}, {}, []
    for seed in SEEDS:
        basis = make_basis(state, seed, 1, rank, (0.1,))
        original, augmented = sufficient_stats(state, basis, x, beta, structures, True)
        for joint in structures:
            for eta in ETAS:
                solved, info = solve_stats(
                    combine(original[joint], augmented[joint], eta), metrics, ALPHAS
                )
                operations.append(dict(seed=seed, joint=joint, eta=eta, **info))
                for (base, alpha), theta in solved.items():
                    family = f"{base}_{'joint_soft' if joint else 'static'}"
                    name = model_id(family, seed, alpha, eta)
                    models[name] = export_model(state, basis, theta, beta, joint)
                    metadata[name] = dict(
                        family=family, seed=seed, alpha=alpha, eta=eta, fallback=False
                    )
        if beta is None:
            for base in BASES:
                for eta in ETAS:
                    for alpha in ALPHAS:
                        family = f"{base}_joint_soft"
                        name = model_id(family, seed, alpha, eta)
                        models[name] = dict(models[model_id(f"{base}_static", seed, alpha, eta)])
                        metadata[name] = dict(
                            family=family, seed=seed, alpha=alpha, eta=eta, fallback=True
                        )
    assert len(models) == 144
    return models, dict(
        models=metadata,
        operations=operations,
        fit_bank_seconds=time.perf_counter() - started,
        new_coefficient_solutions=sum(len(r["solutions"]) for r in operations),
        gram_decompositions=len(operations),
        basis_preparations=3,
        auxiliary_baseline_solves=3,
        auxiliary_theta_solves=3,
        gate_fits=int(beta is not None),
        n_fit_rows=len(x),
        original_weight_mass=float(state["weights"].sum()),
        augmented_copies_per_source=16,
    )


def fit_single(x, y, person, site, camera, family, seed, alpha, eta, rank=128):
    if family not in FAMILIES or alpha not in ALPHAS or eta not in ETAS:
        raise ValueError("registered family and alpha/eta required")
    started = time.perf_counter()
    base, structure = family.split("_", 1)
    joint = structure == "joint_soft"
    state = prepare(x, y, weights_for(person, site))
    beta = fit_gate(state["normalized"], person, site, camera) if joint else None
    active = joint and beta is not None
    basis = make_basis(state, seed, 1, rank, (0.1,))
    original, augmented = sufficient_stats(state, basis, x, beta, (active,), eta > 0)
    solved, info = solve_stats(
        combine(original[active], augmented[active], eta), color_metrics(state, (base,)), (alpha,)
    )
    model = export_model(state, basis, solved[(base, alpha)], beta, active)
    return model, dict(
        fit_seconds=time.perf_counter() - started,
        active_gate=active,
        synthetic_fit_rows=len(x) * 16 if eta > 0 else 0,
        **info,
    )
