"""Independent C teacher geometry/QR and native-unit residual reconstruction."""

from __future__ import annotations

import numpy as np
from chromaseed_affine_reference import qr_ridge
from chromaseed_projection_reference import refit


def teacher(template, x, y, person, site, camera, loss, seed, rank=128):
    return refit(template, x, y, person, site, camera, loss + "_joint_soft", 0.1, "raw", seed, rank)


def head(basis, native_teacher, loss, setting):
    g = basis.g
    native = np.asarray(native_teacher, np.float64)
    if native.shape != g.y.shape or not np.isfinite(native).all():
        raise ValueError("one native-Lab teacher prediction per original student row required")
    residual = (g.y.astype(np.float64) - native) / g.prep["y_std"]
    design = basis.pd
    if setting["kind"] == "support":
        h = np.sum(g.kr[:, basis.ids], axis=1) / len(basis.ids)
        design = h[:, None] ** setting["power"] * design
    theta = qr_ridge(design, residual, g.w, g.metrics[loss], setting["alpha"])
    branch, raw = basis.export(theta, True), basis.raw[loss]
    if setting["rho"] == 0:
        return dict(raw)
    result = dict(
        **raw,
        latent_projection=branch["projection"],
        latent_mean=branch["projection_mean"],
        latent_width=branch["width"],
        latent_coefficient=branch["coefficient"],
        hybrid_mode=np.array(3 if setting["kind"] == "support" else 2, np.uint8),
        mix=np.array(setting["rho"], np.float32),
        support_power=np.array(setting["power"], np.uint8),
    )
    if "correction" in raw:
        result["latent_correction"] = branch["correction"]
    return result
