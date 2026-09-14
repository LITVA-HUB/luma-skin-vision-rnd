"""Independent dense distances, weighted SVD geometry and QR/SVD readouts for H."""

from __future__ import annotations

import numpy as np
from chromaseed_affine_reference import qr_ridge
from chromaseed_gated_audit import reference_gate
from chromaseed_kernel import select_landmarks
from chromaseed_kernel_audit import direct_kernel, norm
from chromaseed_perceptual_audit import balanced
from chromaseed_perceptual_reference import analytic_tensor
from chromaseed_projection_reference import direct_width, projector, spectrum
from chromaseed_projection_reference import predict as original_predict


def signal(model, x):
    if "gate_beta" not in model:
        return None
    return np.clip(
        np.sum(norm(model, x) * model["gate_beta"][None, 1:].astype(np.float64), axis=1)
        + float(model["gate_beta"][0]),
        -1,
        1,
    )


def predict(model, x):
    if "hybrid_mode" not in model:
        return original_predict(model, x)
    raw = norm(model, x)
    p, m = model["latent_projection"].astype(np.float64), model["latent_mean"].astype(np.float64)
    q = ((raw - m) @ p).astype(np.float32).astype(np.float64)
    centers = ((model["centers"].astype(np.float64) - m) @ p).astype(np.float32)
    kr = direct_kernel(raw, model["centers"], float(model["width"]))
    kp = direct_kernel(q, centers, float(model["latent_width"]))
    base, branch = (
        kr @ model["coefficient"].astype(np.float64),
        kp @ model["latent_coefficient"].astype(np.float64),
    )
    s = signal(model, x)
    if s is not None:
        base += s[:, None] * (kr @ model["correction"].astype(np.float64))
        branch += s[:, None] * (kp @ model["latent_correction"].astype(np.float64))
    mix = float(model["mix"])
    if int(model["hybrid_mode"]) == 1:
        value = base * (1 - mix) + mix * branch
    else:
        cover = (np.sum(kr, axis=1) / kr.shape[1]) ** int(model["support_power"])
        value = base + mix * cover[:, None] * branch
    return value * model["y_std"] + model["y_mean"]


def whitening(km):
    u, s, _ = np.linalg.svd(km, full_matrices=False)
    keep = s > 1e-8 * s.max()
    return u[:, keep] / np.sqrt(s[keep])


class Geometry:
    def __init__(self, x, y, person, site, camera):
        self.x, self.y, self.w = x, y, balanced(person, site)
        self.prep = {}
        for prefix, a in (("x", x), ("y", y)):
            a = a.astype(np.float64)
            self.prep[prefix + "_mean"] = a.mean(0).astype(np.float32)
            self.prep[prefix + "_std"] = np.maximum(a.std(0), 1e-6).astype(np.float32)
        self.raw = norm(self.prep, x)
        self.spectrum = spectrum(self.raw, self.w)
        self.projection = projector(self.spectrum, 16, 0.5)
        p, m = (
            self.projection["projection"].astype(np.float64),
            self.projection["projection_mean"].astype(np.float64),
        )
        self.q = ((self.raw - m) @ p).astype(np.float32).astype(np.float64)
        self.rw, self.pw = (float(np.float32(direct_width(v))) for v in (self.raw, self.q))
        self.kr = direct_kernel(self.raw, self.raw, self.rw)
        self.kp = direct_kernel(self.q, self.q, self.pw)
        self.beta = (
            reference_gate(self.raw, person, site, camera) if len(np.unique(camera)) > 1 else None
        )
        self.signal = (
            None
            if self.beta is None
            else np.clip(
                np.sum(self.raw * self.beta[None, 1:].astype(np.float64), axis=1)
                + float(self.beta[0]),
                -1,
                1,
            )
        )
        self.target = (y.astype(np.float64) - self.prep["y_mean"]) / self.prep["y_std"]
        tensor = analytic_tensor(y)
        std = self.prep["y_std"].astype(np.float64)
        tensor = tensor * std[None, :, None] * std[None, None, :]
        tensor /= np.average(np.trace(tensor, axis1=1, axis2=2), weights=self.w) / 3
        self.metrics = dict(norm=np.eye(3), perceptual=np.average(tensor, axis=0, weights=self.w))

    def design(self, z):
        return z if self.signal is None else np.column_stack([z, self.signal[:, None] * z])


class Basis:
    def __init__(self, geometry, seed, rank=128):
        self.g = g = geometry
        self.ids, _, _ = select_landmarks(g.kr, g.w, "rpchol", rank, seed)
        self.rwhite, self.pwhite = (whitening(k[np.ix_(self.ids, self.ids)]) for k in (g.kr, g.kp))
        self.rd, self.pd = (
            g.design(g.kr[:, self.ids] @ self.rwhite),
            g.design(g.kp[:, self.ids] @ self.pwhite),
        )
        self.raw = {}
        for loss, metric in g.metrics.items():
            theta = qr_ridge(self.rd, g.target, g.w, metric, 0.1)
            self.raw[loss] = self.export(theta, False)

    def export(self, theta, projected):
        g = self.g
        white, coords, width = (self.pwhite, g.q, g.pw) if projected else (self.rwhite, g.raw, g.rw)
        d = white.shape[1]
        model = dict(
            **g.prep,
            centers=coords[self.ids].astype(np.float32),
            width=np.array(width, np.float32),
            coefficient=(white @ theta[:d]).astype(np.float32),
        )
        if projected:
            model.update(g.projection)
        if g.beta is not None:
            model.update(
                correction=(white @ theta[d:]).astype(np.float32),
                gate_beta=g.beta,
                gate_mode=np.array(1, np.uint8),
                rho=np.array(1.0, np.float32),
            )
        return model

    def make(self, loss, kind, alpha, rho, power):
        g, raw = self.g, self.raw[loss]
        if kind == "raw" or rho == 0:
            return dict(raw)
        target, d = g.target, self.pd
        if kind in ("uniform", "support"):
            target = target - (predict(raw, g.x) - g.prep["y_mean"]) / g.prep["y_std"]
        if kind == "support":
            cover = np.sum(g.kr[:, self.ids], axis=1) / len(self.ids)
            d = cover[:, None] ** power * d
        theta = qr_ridge(d, target, g.w, g.metrics[loss], alpha)
        branch = self.export(theta, True)
        if kind == "projected" or (kind == "blend" and rho == 1):
            return branch
        m = dict(
            **raw,
            latent_projection=branch["projection"],
            latent_mean=branch["projection_mean"],
            latent_width=branch["width"],
            latent_coefficient=branch["coefficient"],
            hybrid_mode=np.array({"blend": 1, "uniform": 2, "support": 3}[kind], np.uint8),
            mix=np.array(rho, np.float32),
            support_power=np.array(power, np.uint8),
        )
        if g.beta is not None:
            m["latent_correction"] = branch["correction"]
        return m
