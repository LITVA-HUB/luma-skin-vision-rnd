"""Independent augmented-design QR/SVD reference for A; no primary Gram solver."""

from __future__ import annotations

import itertools

import numpy as np
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import reference_gate
from chromaseed_kernel_audit import direct_kernel, norm
from chromaseed_perceptual_audit import balanced
from chromaseed_perceptual_reference import analytic_tensor


def qr_ridge(design, target, weights, metric, alpha):
    if alpha <= 0 or np.any(weights <= 0):
        raise ValueError("positive penalty and nonzero stacked weights required")
    d = design.shape[1]
    # Orthogonal compression includes the targets, retaining residual components
    # outside the feature span. This never forms X.T@W@X.
    r = np.linalg.qr(np.column_stack([design, target]) * np.sqrt(weights)[:, None], mode="r")
    mu, v = np.linalg.eigh(metric)
    if np.min(mu) <= 0:
        raise ValueError("positive color metric required")
    a, y = r[:, :d], r[:, d:] @ v
    solutions = []
    for j in range(3):
        left = np.concatenate([a, np.sqrt(alpha / mu[j]) * np.eye(d)])
        right = np.r_[y[:, j], np.zeros(d)]
        solutions.append(np.linalg.lstsq(left, right, rcond=None)[0])
    return np.column_stack(solutions) @ v.T


def refit(template, x, y, person, site, camera, family, alpha, eta):
    w = balanced(person, site)
    z = norm(template, x)
    yy = (np.asarray(y, np.float64) - template["y_mean"]) / template["y_std"]
    kmm = direct_kernel(template["centers"], template["centers"], float(template["width"]))
    u, singular, _ = np.linalg.svd(kmm, full_matrices=False)
    keep = singular > 1e-8 * singular.max()
    white = u[:, keep] / np.sqrt(singular[keep])
    joint = family.endswith("joint_soft") and len(np.unique(camera)) > 1
    beta = reference_gate(z, person, site, camera) if joint else None
    samples, targets, masses = [], [], []
    configs = [(0.0, np.zeros(3), 1 - eta)]
    if eta > 0:
        configs += [
            (t, np.array(a), eta / 16)
            for t in (1 / 255, 4 / 255)
            for a in itertools.product((0.0, 1.0), repeat=3)
        ]
    for t, anchor, mass in configs:
        xx = transformed(x, t, anchor)
        zz = norm(template, xx)
        phi = direct_kernel(zz, template["centers"], float(template["width"])) @ white
        if joint:
            s = np.sum(zz * beta[None, 1:].astype(np.float64), axis=1) + float(beta[0])
            phi = np.concatenate([phi, np.clip(s, -1.0, 1.0)[:, None] * phi], axis=1)
        samples.append(phi)
        targets.append(yy)
        masses.append(mass * w)
    x_aug, y_aug, w_aug = np.concatenate(samples), np.concatenate(targets), np.concatenate(masses)
    mass_by_source = w_aug.reshape(len(configs), len(w)).sum(0)
    np.testing.assert_allclose(mass_by_source, w, atol=2e-13, rtol=2e-15)
    metric = np.eye(3)
    if family.startswith("perceptual"):
        raw = analytic_tensor(y)
        std = template["y_std"].astype(np.float64)
        gg = raw * std[None, :, None] * std[None, None, :]
        gg /= np.average(np.trace(gg, axis1=1, axis2=2), weights=w) / 3
        metric = np.average(gg, axis=0, weights=w)
    theta = qr_ridge(x_aug, y_aug, w_aug, metric, alpha)
    dim = white.shape[1]
    result = {
        k: v.copy()
        for k, v in template.items()
        if k in ("x_mean", "x_std", "y_mean", "y_std", "width", "centers")
    }
    result["coefficient"] = (white @ theta[:dim]).astype(np.float32)
    if joint:
        result.update(
            correction=(white @ theta[dim:]).astype(np.float32),
            gate_beta=beta,
            gate_mode=np.array(1, np.uint8),
            rho=np.array(1.0, np.float32),
        )
    return result, dict(
        stacked_rows=len(w_aug),
        source_rows=len(w),
        source_weight_mass_drift=float(np.max(np.abs(mass_by_source - w))),
        retained_dimension=dim,
        active_joint=joint,
    )
