"""Small independent FP64 Adam/Gaussian banks; deployment keeps FP32 means only."""

from __future__ import annotations

import hashlib
import time

import numpy as np
from skin_local_search_train import weights_for

KEYS = ("w1", "b1", "w2", "b2")
GROUPS = {"raw36": list(range(36)), "mean3": [27, 28, 29]}
METHODS = ("adam", "tagi_diag", "tagi_full3")
SEEDS = (17, 29, 43)
CHECKPOINTS = (1, 4, 16, 64)
PARAMETERS = {
    "adam": (0.0003, 0.001, 0.003),
    "tagi_diag": (0.1, 0.3, 1.0),
    "tagi_full3": (0.1, 0.3, 1.0),
}
VAR_FLOOR = 1e-12


def initialize(d, seed, hidden=64):
    if not isinstance(d, int) or not 1 <= d <= 36 or not isinstance(hidden, int) or hidden < 1:
        raise ValueError("positive hidden width and input dimension1..36 required")
    rng = np.random.default_rng(seed)
    mean, variance = {}, {}
    for name, incoming, outgoing in (("1", d, hidden), ("2", hidden, 3)):
        value = 2.0 / (incoming + outgoing)
        mean["w" + name] = rng.normal(0, np.sqrt(value), (outgoing, incoming))
        mean["b" + name] = np.zeros(outgoing, np.float64)
        variance["w" + name] = np.full((outgoing, incoming), value, np.float64)
        variance["b" + name] = np.full(outgoing, 0.01, np.float64)
    return mean, variance


def activations(mean, x):
    hidden = np.einsum("bhd,bd->bh", mean["w1"], x) + mean["b1"]
    active = (hidden > 0).astype(np.float64)
    a = np.maximum(hidden, 0)
    output = np.einsum("boh,bh->bo", mean["w2"], a) + mean["b2"]
    return a, active, output


def gaussian_step(mean, variance, x, y, weight, sigma, kind):
    if kind not in ("tagi_diag", "tagi_full3") or np.any(weight <= 0) or np.any(sigma <= 0):
        raise ValueError("valid Gaussian kind, positive weights and noise required")
    a, active, output = activations(mean, x)
    u = active * (np.einsum("bhd,bd->bh", variance["w1"], x * x) + variance["b1"])
    diagonal = np.einsum("boh,bh->bo", variance["w2"], a * a + u) + variance["b2"]
    diagonal += (sigma * sigma / weight)[:, None]
    if kind == "tagi_full3":
        innovation = np.einsum("boh,bh,bph->bop", mean["w2"], u, mean["w2"])
        idx = np.arange(3)
        innovation[:, idx, idx] += diagonal
        inverse = np.linalg.inv(innovation)
        inverse_diag = np.diagonal(inverse, axis1=1, axis2=2)
        q = np.einsum("bop,bp->bo", inverse, y - output)
        curvature = np.einsum("boh,bop,bph->bh", mean["w2"], inverse, mean["w2"])
    else:
        diagonal += np.einsum("boh,bh->bo", mean["w2"] ** 2, u)
        inverse_diag = 1.0 / diagonal
        q = inverse_diag * (y - output)
        curvature = np.einsum("boh,bo->bh", mean["w2"] ** 2, inverse_diag)
    # All reverse quantities use the same pre-update W2 and hidden moments.
    g = active * np.einsum("boh,bo->bh", mean["w2"], q)
    h = active * curvature
    factors = {
        "w1": (x[:, None, :], g[:, :, None], h[:, :, None]),
        "b1": (1.0, g, h),
        "w2": (a[:, None, :], q[:, :, None], inverse_diag[:, :, None]),
        "b2": (1.0, q, inverse_diag),
    }
    stats = dict(
        floored=np.zeros(len(x), np.int64),
        negative=np.zeros(len(x), np.int64),
        min_pre=np.full(len(x), np.inf),
    )
    for key in KEYS:
        feature, delta, precision = factors[key]
        cross = variance[key] * feature
        mean[key] += cross * delta
        pre = variance[key] - cross * cross * precision
        flat = pre.reshape(len(x), -1)
        stats["floored"] += np.sum(flat < VAR_FLOOR, axis=1)
        stats["negative"] += np.sum(flat < 0, axis=1)
        stats["min_pre"] = np.minimum(stats["min_pre"], np.min(flat, axis=1))
        variance[key] = np.maximum(pre, VAR_FLOOR)
    return stats


def adam_step(mean, first, second, x, y, weight, lr, step):
    if step < 1 or np.any(weight <= 0) or np.any(lr <= 0):
        raise ValueError("positive step, weights and learning rates required")
    a, active, output = activations(mean, x)
    error = (output - y) * weight[:, None]
    hidden_error = active * np.einsum("boh,bo->bh", mean["w2"], error)
    gradients = dict(
        w1=hidden_error[:, :, None] * x[:, None, :],
        b1=hidden_error,
        w2=error[:, :, None] * a[:, None, :],
        b2=error,
    )
    for key in KEYS:
        gradient = gradients[key]
        first[key] *= 0.9
        first[key] += 0.1 * gradient
        second[key] *= 0.999
        second[key] += 0.001 * gradient * gradient
        rate = lr.reshape((-1,) + (1,) * (gradient.ndim - 1))
        denominator = np.sqrt(second[key]) / np.sqrt(1.0 - 0.999**step) + 1e-8
        mean[key] -= rate * (first[key] / (1.0 - 0.9**step)) / denominator


def prepare(x, y, group):
    x, y = np.asarray(x), np.asarray(y)
    if (
        group not in GROUPS
        or x.ndim != 2
        or x.shape[1] != 36
        or len(x) == 0
        or y.shape != (len(x), 3)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("nonempty finite color36/native Lab3 and declared group required")
    selected = x[:, GROUPS[group]].astype(np.float64)
    target = y.astype(np.float64)
    prep = dict(
        x_mean=selected.mean(0).astype(np.float32),
        x_std=np.maximum(selected.std(0), 1e-6).astype(np.float32),
        y_mean=target.mean(0).astype(np.float32),
        y_std=np.maximum(target.std(0), 1e-6).astype(np.float32),
    )
    if group != "raw36":
        prep["feature_indices"] = np.array(GROUPS[group], np.uint8)
    xn = ((selected.astype(np.float32) - prep["x_mean"]) / prep["x_std"]).astype(np.float64)
    yn = ((target.astype(np.float32) - prep["y_mean"]) / prep["y_std"]).astype(np.float64)
    return prep, xn, yn


def export(prep, mean):
    return {
        **{k: v.copy() for k, v in prep.items()},
        **{k: mean[k].astype(np.float32) for k in KEYS},
    }


def predict(model, x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
        raise ValueError("finite N by36 input required")
    if "feature_indices" in model:
        x = x[:, model["feature_indices"]]
    z = ((x - model["x_mean"]) / model["x_std"]).astype(np.float64)
    hidden = np.maximum(z @ model["w1"].astype(np.float64).T + model["b1"].astype(np.float64), 0)
    value = hidden @ model["w2"].astype(np.float64).T + model["b2"].astype(np.float64)
    return value * model["y_std"] + model["y_mean"]


def train_block(x, y, weights, group, cases, checkpoints=CHECKPOINTS, hidden=64):
    started = time.perf_counter()
    prep, xn, yn = prepare(x, y, group)
    weights = np.asarray(weights, np.float64)
    if weights.shape != (len(xn),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("finite positive row weights required")
    if (
        not cases
        or any(
            c["method"] not in METHODS or not np.isfinite(c["parameter"]) or c["parameter"] <= 0
            for c in cases
        )
        or len({c["method"] for c in cases}) != 1
    ):
        raise ValueError("one valid method and independent cases required per block")
    if (
        not checkpoints
        or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
        or any(not isinstance(e, int) or e < 1 for e in checkpoints)
    ):
        raise ValueError("strictly increasing positive integer checkpoints required")
    method, count = cases[0]["method"], len(cases)
    initial = [initialize(xn.shape[1], int(c["seed"]), hidden) for c in cases]
    mean = {k: np.stack([pair[0][k] for pair in initial]) for k in KEYS}
    state = {k: np.stack([pair[1][k] for pair in initial]) for k in KEYS}
    del initial
    if method == "adam":
        state = {k: np.zeros_like(v) for k, v in mean.items()}
        second = {k: np.zeros_like(v) for k, v in mean.items()}
    parameters = np.array([c["parameter"] for c in cases], np.float64)
    generators = [np.random.default_rng(int(c["seed"]) + 600017) for c in cases]
    digests = [hashlib.sha256() for _ in cases]
    counters = dict(
        floored=np.zeros(count, np.int64),
        negative=np.zeros(count, np.int64),
        min_pre=np.full(count, np.inf),
    )
    parameter_bytes = sum(v.nbytes for v in mean.values()) // count
    records = [
        dict(
            **case,
            group=group,
            n_fit_rows=len(xn),
            parameter_count=parameter_bytes // 8,
            training_parameter_state_bytes=parameter_bytes * (3 if method == "adam" else 2),
            checkpoints=[],
        )
        for case in cases
    ]
    models, steps = {}, 0
    for epoch in range(1, max(checkpoints) + 1):
        order = np.stack([rng.permutation(len(xn)) for rng in generators])
        for digest, indices in zip(digests, order, strict=True):
            digest.update(np.asarray(indices, np.int64).tobytes())
        xx, yy, ww = xn[order], yn[order], weights[order]
        for pos in range(len(xn)):
            steps += 1
            if method == "adam":
                adam_step(
                    mean, state, second, xx[:, pos], yy[:, pos], ww[:, pos], parameters, steps
                )
            else:
                stats = gaussian_step(
                    mean, state, xx[:, pos], yy[:, pos], ww[:, pos], parameters, method
                )
                counters["floored"] += stats["floored"]
                counters["negative"] += stats["negative"]
                counters["min_pre"] = np.minimum(counters["min_pre"], stats["min_pre"])
        if not all(np.isfinite(v).all() for obj in (mean, state) for v in obj.values()):
            raise FloatingPointError(f"Non-finite {method}/{group} fit state at epoch{epoch}")
        if method == "adam" and not all(np.isfinite(v).all() for v in second.values()):
            raise FloatingPointError("Non-finite Adam second moment")
        if epoch in checkpoints:
            elapsed = time.perf_counter() - started
            for i in range(count):
                model = export(prep, {k: v[i] for k, v in mean.items()})
                if not all(np.isfinite(v).all() for v in model.values()):
                    raise FloatingPointError("Non-finite FP32 exported parameter")
                models[(i, epoch)] = model
                records[i]["checkpoints"].append(
                    dict(
                        epoch=epoch,
                        examples_seen=steps,
                        order_sha256=digests[i].hexdigest(),
                        block_elapsed_seconds=elapsed,
                        floored_variance_updates=int(counters["floored"][i]),
                        negative_variance_updates=int(counters["negative"][i]),
                        minimum_pre_floor_variance=None
                        if method == "adam"
                        else float(counters["min_pre"][i]),
                        minimum_variance=None
                        if method == "adam"
                        else min(float(v[i].min()) for v in state.values()),
                        maximum_abs_mean=max(float(np.abs(v[i]).max()) for v in mean.values()),
                    )
                )
    return models, records


def fit_single(x, y, person, site, group, method, parameter, seed, epochs):
    weights = weights_for(person, site)
    models, records = train_block(
        x, y, weights, group, [dict(method=method, parameter=parameter, seed=seed)], (epochs,)
    )
    return models[(0, epochs)], records[0]


def model_id(method, group, seed, parameter, epoch):
    return f"{method}_{group}_s{seed}_h{PARAMETERS[method].index(parameter)}_e{epoch}"


def choose(candidates):
    if not candidates or any(
        not np.isfinite(c["clean"]) or not np.isfinite(c["p90"]) for c in candidates
    ):
        raise ValueError("finite nonempty candidate scores required")
    return min(candidates, key=lambda c: (c["clean"], c["p90"], c["epoch"], c["parameter_index"]))
