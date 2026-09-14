"""NumPy-only one-record consumer; projected centers are derived once at load."""

from __future__ import annotations

import numpy as np
from chromaseed_projection_numpy import Predictor as OriginalPredictor


class Predictor(OriginalPredictor):
    def __init__(self, model):
        self.hybrid = "hybrid_mode" in model
        if not self.hybrid:
            super().__init__(model)
            return
        extra = {
            "latent_projection",
            "latent_mean",
            "latent_width",
            "latent_coefficient",
            "hybrid_mode",
            "mix",
            "support_power",
        }
        if "correction" in model:
            extra.add("latent_correction")
        if not extra <= set(model):
            raise ValueError("incomplete hybrid payload")
        super().__init__({k: v for k, v in model.items() if k not in extra})
        if self.projection is not None or self.constant is not None:
            raise ValueError("raw-center backbone required")
        n = len(self.centers)
        shape = dict(
            latent_projection=(36, 16),
            latent_mean=(36,),
            latent_width=(),
            latent_coefficient=(n, 3),
            hybrid_mode=(),
            mix=(),
            support_power=(),
        )
        if self.correction is not None:
            shape["latent_correction"] = (n, 3)
        for k, s in shape.items():
            dtype = np.uint8 if k in ("hybrid_mode", "support_power") else np.float32
            if model[k].shape != s or model[k].dtype != dtype or not np.isfinite(model[k]).all():
                raise ValueError("invalid hybrid shape/dtype/value")
        self.mode, self.mix, self.power = (
            int(model["hybrid_mode"]),
            float(model["mix"]),
            int(model["support_power"]),
        )
        if self.mode not in (1, 2, 3) or self.mix not in (0.25, 0.5, 1.0):
            raise ValueError("registered positive mixture required")
        if (
            self.power not in ((1, 4) if self.mode == 3 else (0,))
            or float(model["latent_width"]) <= 0
        ):
            raise ValueError("valid support and width required")
        self.lp, self.lm = (
            model["latent_projection"].astype(np.float64),
            model["latent_mean"].astype(np.float64),
        )
        self.lcenters = ((self.centers - self.lm) @ self.lp).astype(np.float32).astype(np.float64)
        self.lnorms = np.sum(self.lcenters * self.lcenters, axis=1)
        self.ldenom = 16 * float(model["latent_width"]) ** 2
        self.lcoef = model["latent_coefficient"].astype(np.float64)
        self.lcorr = (
            None if self.correction is None else model["latent_correction"].astype(np.float64)
        )

    def __call__(self, color):
        if not self.hybrid:
            return super().__call__(color)
        color = np.asarray(color, np.float32)
        if color.shape != (36,) or not np.isfinite(color).all():
            raise ValueError("one finite color36 vector required")
        z = ((color - self.x_mean) / self.x_std).astype(np.float64)
        distance = np.maximum(np.sum(z * z) + self.norms - 2 * (self.centers @ z), 0)
        raw_k = np.exp(-0.5 * distance / self.denominator)
        q = ((z - self.lm) @ self.lp).astype(np.float32).astype(np.float64)
        ld = np.maximum(np.sum(q * q) + self.lnorms - 2 * (self.lcenters @ q), 0)
        k = np.exp(-0.5 * ld / self.ldenom)
        base, branch = raw_k @ self.coefficient, k @ self.lcoef
        if self.correction is not None:
            signal = min(max(float(z @ self.beta[1:] + self.beta[0]), -1), 1)
            base += signal * (raw_k @ self.correction)
            branch += signal * (k @ self.lcorr)
        if self.mode == 1:
            value = (1 - self.mix) * base + self.mix * branch
        else:
            scale = float(raw_k.mean()) ** self.power if self.mode == 3 else 1.0
            value = base + self.mix * scale * branch
        return value * self.y_std + self.y_mean
