"""Analytical output fits folded into unchanged small TG networks."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_gaussian import (
    export,
    initialize,
    predict,
    prepare,
    train_block,
)
from chromaseed_gaussian import (
    fit_single as tg_fit,
)
from chromaseed_gaussian_numpy import Predictor
from chromaseed_perceptual import local_tensor
from scipy.linalg import cho_factor, cho_solve
from skin_local_search_train import weights_for

SEEDS = (17, 29, 43)
EPOCHS = (1, 4, 16)
BASES = (
    "random",
    "adam_e1",
    "adam_e4",
    "adam_e16",
    "tagi_full3_e1",
    "tagi_full3_e4",
    "tagi_full3_e16",
)
FAMILIES = ("norm", "perceptual")
ALPHAS = (0.1, 1.0, 10.0)
PARAMETERS = {"adam": 0.001, "tagi_full3": 1.0}
GROUPS = ("raw36", "mean3")


def basis_spec(basis):
    if basis == "random":
        return "random", 0
    method, epoch = basis.rsplit("_e", 1)
    if method not in PARAMETERS or int(epoch) not in EPOCHS:
        raise ValueError("registered representation required")
    return method, int(epoch)


def solve(z, target, metric, weights, alpha):
    z, target, weights = (np.asarray(v, np.float64) for v in (z, target, weights))
    if (
        z.ndim != 2
        or not len(z)
        or z.shape[1] < 1
        or target.shape != (len(z), 3)
        or weights.shape != (len(z),)
        or not np.isfinite(alpha)
        or alpha <= 0
        or np.any(weights <= 0)
        or not all(np.isfinite(v).all() for v in (z, target, weights))
        or not np.array_equal(z[:, 0], np.ones(len(z)))
    ):
        raise ValueError(
            "finite nonempty design with intercept, Lab3, positive weights and alpha required"
        )
    rank = z.shape[1]
    if metric is None:
        system = z.T @ (weights[:, None] * z)
        system.flat[:: rank + 1] += np.r_[0.0, np.full(rank - 1, alpha)]
        rhs = z.T @ (weights[:, None] * target)
    else:
        metric = np.asarray(metric, np.float64)
        if (
            metric.shape != (len(z), 3, 3)
            or not np.isfinite(metric).all()
            or not np.allclose(metric, metric.transpose(0, 2, 1), atol=1e-12)
        ):
            raise ValueError("finite symmetric per-row output metric required")
        if np.min(np.linalg.eigvalsh(metric)) <= 0:
            raise ValueError("positive definite metric required")
        system = np.empty((3 * rank, 3 * rank))
        for a in range(3):
            for b in range(a, 3):
                block = z.T @ ((weights * metric[:, a, b])[:, None] * z)
                system[a::3, b::3], system[b::3, a::3] = block, block.T
        system.flat[:: 3 * rank + 1] += np.r_[np.zeros(3), np.full(3 * (rank - 1), alpha)]
        rhs = (z.T @ (weights[:, None] * np.einsum("nij,nj->ni", metric, target))).ravel()
    answer = cho_solve(cho_factor(system, lower=True, check_finite=False), rhs, check_finite=False)
    residual = float(np.linalg.norm(system @ answer - rhs) / max(np.linalg.norm(rhs), 1.0))
    if not np.isfinite(answer).all():
        raise FloatingPointError("nonfinite analytical solution")
    return answer.reshape(rank, 3), residual


def fit_head(source, x, y, weights, family, alpha):
    started = time.perf_counter()
    Predictor(source)
    x, y, weights = (
        np.asarray(x, np.float32),
        np.asarray(y, np.float64),
        np.asarray(weights, np.float64),
    )
    if (
        x.ndim != 2
        or x.shape[1] != 36
        or len(x) == 0
        or y.shape != (len(x), 3)
        or weights.shape != (len(x),)
        or family not in FAMILIES
        or not all(np.isfinite(v).all() for v in (x, y, weights))
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "finite color36/native Lab3, positive row weights and registered loss required"
        )
    selected = x[:, source["feature_indices"]] if "feature_indices" in source else x
    xn = ((selected - source["x_mean"]) / source["x_std"]).astype(np.float64)
    hidden = np.maximum(xn @ source["w1"].astype(np.float64).T + source["b1"].astype(np.float64), 0)
    mean, raw_std = hidden.mean(0), hidden.std(0)
    std = np.maximum(raw_std, 1e-6)
    design = np.column_stack((np.ones(len(x)), (hidden - mean) / std))
    target = (y - source["y_mean"]) / source["y_std"]
    metric, scale, tensor_info = None, None, None
    if family == "perceptual":
        tensor, tensor_info = local_tensor(y)
        ys = source["y_std"].astype(np.float64)
        unscaled = tensor * ys[None, :, None] * ys[None, None, :]
        scale = float(np.average(np.trace(unscaled, axis1=1, axis2=2), weights=weights) / 3)
        metric = unscaled / scale
    beta, residual = solve(design, target, metric, weights, alpha)
    folded = beta[1:] / std[:, None]
    model = {k: v.copy() for k, v in source.items()}
    model["w2"] = folded.T.astype(np.float32)
    model["b2"] = (beta[0] - mean @ folded).astype(np.float32)
    reference = (design @ beta) * source["y_std"] + source["y_mean"]
    drift = float(np.max(abs(predict(model, x) - reference)))
    if not all(np.isfinite(v).all() for v in model.values()) or drift > 0.001:
        raise FloatingPointError(f"invalid folded FP32 head, native Lab drift={drift}")
    return model, dict(
        fit_seconds=time.perf_counter() - started,
        normal_residual=residual,
        max_folded_lab_drift=drift,
        hidden_constant_columns=int(np.sum(raw_std < 1e-6)),
        metric_scale=scale,
        tensor_info=tensor_info,
        linear_system_dimension=design.shape[1] * (3 if family == "perceptual" else 1),
        numeric_bytes=sum(v.nbytes for v in model.values()),
        temporary_design_bytes=design.nbytes,
        temporary_metric_bytes=0 if metric is None else metric.nbytes,
        temporary_system_bytes=(design.shape[1] * (3 if family == "perceptual" else 1)) ** 2 * 8,
    )


def representation_bank(x, y, weights, group, seeds=SEEDS, epochs=EPOCHS):
    prep, xn, _ = prepare(x, y, group)
    bases = {("random", seed): export(prep, initialize(xn.shape[1], seed)[0]) for seed in seeds}
    records = []
    for method, parameter in PARAMETERS.items():
        cases = [dict(method=method, parameter=parameter, seed=seed) for seed in seeds]
        models, rec = train_block(x, y, weights, group, cases, checkpoints=epochs)
        for (i, epoch), model in models.items():
            bases[(f"{method}_e{epoch}", seeds[i])] = model
        records.extend(rec)
    return bases, records


def fit_single(x, y, person, site, group, basis, family, alpha, seed):
    started = time.perf_counter()
    weights = weights_for(person, site)
    method, epoch = basis_spec(basis)
    if method == "random":
        prep, xn, _ = prepare(x, y, group)
        source = export(prep, initialize(xn.shape[1], seed)[0])
        training = None
    else:
        source, training = tg_fit(
            x, y, person, site, group, method, PARAMETERS[method], seed, epoch
        )
    if family == "unchanged":
        model, head = source, None
    else:
        model, head = fit_head(source, x, y, weights, family, alpha)
    return model, dict(
        total_seconds=time.perf_counter() - started, representation=training, readout=head
    )


def name_for(family, basis, group, seed, ai=None):
    base = f"{family}_{basis}_{group}_s{seed}"
    return base if ai is None else f"{base}_a{ai}"


def choose(candidates):
    return min(candidates, key=lambda c: (c["clean"], c["p90"], c["alpha_index"]))


def choose_policy(candidates):
    return min(
        candidates,
        key=lambda c: (c["clean"], c["p90"], basis_spec(c["basis"])[1], BASES.index(c["basis"])),
    )
