"""Independent weighted-design SVD projection, dense kernel and QR/SVD refit."""

from __future__ import annotations

import numpy as np
from chromaseed_affine_reference import qr_ridge
from chromaseed_gated_audit import reference_gate
from chromaseed_kernel import select_landmarks
from chromaseed_kernel_audit import direct_kernel, norm
from chromaseed_perceptual_audit import balanced
from chromaseed_perceptual_reference import analytic_tensor


def spectrum(z, w):
    weights = np.asarray(w, np.float64) / np.sum(w)
    mean = np.array([np.dot(weights, c) for c in z.T])
    a = (z - mean) * np.sqrt(weights[:, None])
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    v = vt.T
    for j in range(v.shape[1]):
        if v[np.argmax(np.abs(v[:, j])), j] < 0:
            v[:, j] *= -1
    return dict(
        mean=mean, eigenvalues=s**2, vectors=v, trace=float(np.sum(a * a)), covariance=a.T @ a
    )


def projector(svd, dimension, shrinkage):
    mean_eigen = svd["trace"] / 36
    penalty = (1 - shrinkage) * svd["eigenvalues"][:dimension] + shrinkage * mean_eigen
    denominator = np.sqrt(np.maximum(penalty, 1e-4 * max(mean_eigen, 1e-12)))
    return dict(
        projection_mean=svd["mean"].astype(np.float32),
        projection=(svd["vectors"][:, :dimension] / denominator).astype(np.float32),
    )


def latent(model, x):
    z = norm(model, x)
    if "projection" not in model:
        return z
    # Stored FP64 matrix multiplication followed by the specified FP32 cast.
    return (
        ((z - model["projection_mean"].astype(np.float64)) @ model["projection"].astype(np.float64))
        .astype(np.float32)
        .astype(np.float64)
    )


def direct_width(q):
    values = []
    for i in range(len(q) - 1):
        values.extend(np.sqrt(np.mean((q[i + 1 :] - q[i]) ** 2, axis=1)).tolist())
    ordered = np.sort(np.asarray(values))
    positive = ordered[ordered > 1e-10]
    return max(float(positive[(len(positive) - 1) // 2]), 1e-6) if len(positive) else 1e-6


def predict(model, x):
    if "constant_lab" in model:
        return np.broadcast_to(model["constant_lab"].astype(np.float64), (len(x), 3)).copy()
    k = direct_kernel(latent(model, x), model["centers"], float(model["width"]))
    value = k @ model["coefficient"].astype(np.float64)
    if "correction" in model:
        raw = norm(model, x)
        signal = np.sum(raw * model["gate_beta"][None, 1:].astype(np.float64), axis=1) + float(
            model["gate_beta"][0]
        )
        value += (
            float(model["rho"])
            * np.clip(signal, -1, 1)[:, None]
            * (k @ model["correction"].astype(np.float64))
        )
    return value * model["y_std"] + model["y_mean"]


def refit(template, x, y, person, site, camera, family, alpha, representation, seed, rank=128):
    w = balanced(person, site)
    m = {}
    for prefix, a in (("x", x), ("y", y)):
        a = a.astype(np.float64)
        m[prefix + "_mean"] = a.mean(0).astype(np.float32)
        m[prefix + "_std"] = np.maximum(a.std(0), 1e-6).astype(np.float32)
    raw = norm(m, x)
    if representation != "raw":
        d, label = representation.split("_t")
        tau = {"0": 0.0, "01": 0.1, "05": 0.5, "1": 1.0}[label]
        m.update(projector(spectrum(raw, w), int(d[1:]), tau))
    q = latent(m, x)
    width = float(np.float32(direct_width(q)))
    kernel = direct_kernel(q, q, width)
    ids, _, _ = select_landmarks(kernel, w, "rpchol", rank, seed)
    km = kernel[np.ix_(ids, ids)]
    u, s, _ = np.linalg.svd(km, full_matrices=False)
    keep = s > 1e-8 * s.max()
    white = u[:, keep] / np.sqrt(s[keep])
    design = kernel[:, ids] @ white
    joint = family.endswith("joint_soft") and len(np.unique(camera)) > 1
    beta = reference_gate(raw, person, site, camera) if joint else None
    if joint:
        score = np.sum(raw * beta[None, 1:].astype(np.float64), axis=1) + float(beta[0])
        design = np.column_stack([design, np.clip(score, -1, 1)[:, None] * design])
    target = (np.asarray(y, np.float64) - m["y_mean"]) / m["y_std"]
    metric = np.eye(3)
    if family.startswith("perceptual"):
        raw_metric = analytic_tensor(y)
        std = m["y_std"].astype(np.float64)
        g = raw_metric * std[None, :, None] * std[None, None, :]
        g /= np.average(np.trace(g, axis1=1, axis2=2), weights=w) / 3
        metric = np.average(g, axis=0, weights=w)
    theta = qr_ridge(design, target, w, metric, alpha)
    dim = white.shape[1]
    m.update(
        centers=q[ids].astype(np.float32),
        width=np.array(width, np.float32),
        coefficient=(white @ theta[:dim]).astype(np.float32),
    )
    if joint:
        m.update(
            correction=(white @ theta[dim:]).astype(np.float32),
            gate_beta=beta,
            gate_mode=np.array(1, np.uint8),
            rho=np.array(1.0, np.float32),
        )
    return m, dict(
        fit_center_indices=ids.tolist(),
        actual_centers=len(ids),
        effective_rank=dim,
        width_drift=abs(width - float(template["width"])),
        active_gate=joint,
        projection_max_difference=0.0
        if "projection" not in m
        else float(np.abs(m["projection"] - template["projection"]).max()),
    )
