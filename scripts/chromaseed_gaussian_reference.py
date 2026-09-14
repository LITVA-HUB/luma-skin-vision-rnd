"""Independent scalar Gaussian smoother / Adam reference; no primary-core import."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.linalg import cho_solve

KEYS = ("w1", "b1", "w2", "b2")


def initial(d, seed, hidden=64):
    rng = np.random.default_rng(seed)
    v1, v2 = 2.0 / (d + hidden), 2.0 / (hidden + 3)
    mean = dict(
        w1=rng.standard_normal((hidden, d)) * np.sqrt(v1),
        b1=np.zeros(hidden),
        w2=rng.standard_normal((3, hidden)) * np.sqrt(v2),
        b2=np.zeros(3),
    )
    var = dict(
        w1=np.full((hidden, d), v1),
        b1=np.full(hidden, 0.01),
        w2=np.full((3, hidden), v2),
        b2=np.full(3, 0.01),
    )
    return mean, var


def normalization(x, y, group):
    if group not in ("raw36", "mean3"):
        raise ValueError("declared feature group required")
    coordinates = np.arange(36) if group == "raw36" else np.array([27, 28, 29])
    z, t = np.asarray(x[:, coordinates], np.float64), np.asarray(y, np.float64)
    prep = {}
    for key, values in (("x", z), ("y", t)):
        mu = np.sum(values, axis=0) / len(values)
        sigma = np.sqrt(np.sum((values - mu) ** 2, axis=0) / len(values))
        prep[key + "_mean"] = np.asarray(mu, np.float32)
        prep[key + "_std"] = np.asarray(np.maximum(sigma, 1e-6), np.float32)
    if group == "mean3":
        prep["feature_indices"] = coordinates.astype(np.uint8)
    return (
        prep,
        ((z.astype(np.float32) - prep["x_mean"]) / prep["x_std"]).astype(np.float64),
        ((t.astype(np.float32) - prep["y_mean"]) / prep["y_std"]).astype(np.float64),
    )


def gaussian_update(mean, variance, x, y, weight, sigma, kind):
    """Condition hidden states first, then incoming parameters by their local gains."""
    m_hidden = mean["w1"] @ x + mean["b1"]
    v_hidden = variance["w1"] @ np.square(x) + variance["b1"]
    slope = (m_hidden > 0).astype(float)
    a = np.clip(m_hidden, 0, None)
    v_a = slope * slope * v_hidden
    expected = mean["w2"] @ a + mean["b2"]
    cross_hidden = (v_hidden * slope)[:, None] * mean["w2"].T
    innovation = (mean["w2"] * v_a[None, :]) @ mean["w2"].T
    extra = variance["w2"] @ (np.square(a) + v_a) + variance["b2"] + sigma * sigma / weight
    innovation = innovation + np.diag(extra)
    if kind == "tagi_diag":
        innovation = np.diag(np.diag(innovation))
    elif kind != "tagi_full3":
        raise ValueError("unknown Gaussian variant")
    lower = np.linalg.cholesky(innovation)
    solved_error = cho_solve((lower, True), y - expected, check_finite=False)
    solved_cross = cho_solve((lower, True), cross_hidden.T, check_finite=False)
    inverse_diag = np.diag(cho_solve((lower, True), np.eye(3), check_finite=False))
    posterior_hidden_mean = m_hidden + cross_hidden @ solved_error
    posterior_hidden_var = v_hidden - np.sum(cross_hidden * solved_cross.T, axis=1)
    new_mean, new_var = {}, {}
    for key, features in (("w1", x), ("b1", np.array(1.0))):
        covariance = variance[key] * features
        denominator = v_hidden[:, None] if key == "w1" else v_hidden
        gain = np.divide(
            covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0
        )
        change_m = posterior_hidden_mean - m_hidden
        change_v = posterior_hidden_var - v_hidden
        if key == "w1":
            change_m, change_v = change_m[:, None], change_v[:, None]
        new_mean[key] = mean[key] + gain * change_m
        new_var[key] = variance[key] + gain * gain * change_v
    for key, features in (("w2", a), ("b2", np.array(1.0))):
        covariance = variance[key] * features
        innovation_mean = solved_error[:, None] if key == "w2" else solved_error
        precision = inverse_diag[:, None] if key == "w2" else inverse_diag
        new_mean[key] = mean[key] + covariance * innovation_mean
        new_var[key] = variance[key] - covariance * covariance * precision
    flat = np.concatenate([new_var[k].ravel() for k in KEYS])
    stats = dict(
        floored=int(np.sum(flat < 1e-12)),
        negative=int(np.sum(flat < 0)),
        min_pre=float(np.min(flat)),
    )
    for key in KEYS:
        mean[key] = new_mean[key]
        variance[key] = np.clip(new_var[key], 1e-12, None)
    return stats


def adam_update(mean, first, second, x, y, weight, lr, step):
    pre = np.dot(mean["w1"], x) + mean["b1"]
    hidden = np.maximum(0, pre)
    prediction = np.dot(mean["w2"], hidden) + mean["b2"]
    delta = weight * (prediction - y)
    hidden_delta = np.dot(mean["w2"].T, delta) * (pre > 0)
    gradients = {
        "w1": np.outer(hidden_delta, x),
        "b1": hidden_delta,
        "w2": np.outer(delta, hidden),
        "b2": delta,
    }
    for key in KEYS:
        first[key] = 0.9 * first[key] + 0.1 * gradients[key]
        second[key] = 0.999 * second[key] + 0.001 * np.square(gradients[key])
        corrected_m = first[key] / (1 - 0.9**step)
        corrected_v = second[key] / (1 - 0.999**step)
        mean[key] = mean[key] - lr * corrected_m / (np.sqrt(corrected_v) + 1e-8)


def train_reference(
    x, y, weights, group, method, parameter, seed, checkpoints=(1, 4, 16, 64), hidden=64
):
    prep, xn, yn = normalization(x, y, group)
    mean, state = initial(xn.shape[1], seed, hidden)
    if method == "adam":
        state = {k: np.zeros_like(v) for k, v in mean.items()}
        second = {k: np.zeros_like(v) for k, v in mean.items()}
    generator, digest = np.random.default_rng(seed + 600017), hashlib.sha256()
    records, models = [], {}
    floored, negative, minimum, step = 0, 0, np.inf, 0
    for epoch in range(1, max(checkpoints) + 1):
        order = generator.permutation(len(xn))
        digest.update(np.asarray(order, np.int64).tobytes())
        for row in order:
            step += 1
            if method == "adam":
                adam_update(mean, state, second, xn[row], yn[row], weights[row], parameter, step)
            else:
                stats = gaussian_update(
                    mean, state, xn[row], yn[row], weights[row], parameter, method
                )
                floored += stats["floored"]
                negative += stats["negative"]
                minimum = min(minimum, stats["min_pre"])
        assert all(np.isfinite(v).all() for obj in (mean, state) for v in obj.values())
        if epoch in checkpoints:
            models[epoch] = {
                **{k: v.copy() for k, v in prep.items()},
                **{k: mean[k].astype(np.float32) for k in KEYS},
            }
            records.append(
                dict(
                    epoch=epoch,
                    examples_seen=step,
                    order_sha256=digest.hexdigest(),
                    floored_variance_updates=floored,
                    negative_variance_updates=negative,
                    minimum_pre_floor_variance=None if method == "adam" else minimum,
                    minimum_variance=None
                    if method == "adam"
                    else min(float(v.min()) for v in state.values()),
                )
            )
    return models, records


def predict(model, x):
    indices = model.get("feature_indices", np.arange(36))
    z = (np.asarray(x, np.float32)[:, indices] - model["x_mean"]) / model["x_std"]
    hidden = np.maximum(
        0, np.matmul(model["w1"].astype(float), z.astype(float).T) + model["b1"][:, None]
    )
    value = np.matmul(model["w2"].astype(float), hidden) + model["b2"][:, None]
    return value.T * model["y_std"] + model["y_mean"]
