"""P3 initialization: change the local encoder, preserve the native HR model."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from chromaseed_architecture_scale import SEEDS, specs
from chromaseed_head_range import Bank as NativeBank
from chromaseed_palette_encoder import ENCODER_PARAMETERS, transplant

VARIANTS = ("patch5m", "soft5m", "dynamic5m")
ARMS = ("original", "aligned", "shuffled")


def encoder_digest(model):
    h = hashlib.sha256()
    for key in sorted(model):
        value = np.asarray(model[key])
        descriptor = f"{key}|{value.dtype.str}|{value.shape}|{value.nbytes}|".encode()
        h.update(len(descriptor).to_bytes(8, "little"))
        h.update(descriptor)
        h.update(value.tobytes())
    return h.hexdigest()


def validate_encoders(encoders, arm):
    if arm not in ARMS:
        raise ValueError("Unregistered palette initialization")
    if arm == "original":
        if encoders is not None:
            raise ValueError("Original initialization must not receive palette encoders")
        return
    if encoders is None or len(encoders) != 3:
        raise ValueError("Three matched pretrained encoders are required")
    for seed, model in zip(SEEDS, encoders, strict=True):
        if set(model) != {"theta", "mean", "std", "seed", "arm"}:
            raise ValueError("Unexpected encoder payload")
        if int(model["seed"]) != seed or str(model["arm"]) != arm:
            raise ValueError("Encoder seed/arm does not match its native slot")
        if (
            model["theta"].shape != (ENCODER_PARAMETERS,)
            or model["theta"].dtype != np.float32
            or model["mean"].shape != (18,)
            or model["std"].shape != (18,)
            or not (model["std"] > 0).all()
            or not all(np.isfinite(model[k]).all() for k in ("theta", "mean", "std"))
        ):
            raise ValueError("Finite complete FP32 encoder with positive scales required")


class Bank(NativeBank):
    def __init__(self, variant, mode, arm, encoders=None, mean=None, std=None):
        if variant not in VARIANTS:
            raise ValueError("P3 requires an architecture with the registered 18→384→256 encoder")
        validate_encoders(encoders, arm)
        assert specs(variant)[:2] == [("token1", 18, 384), ("token2", 384, 256)]
        super().__init__(variant, mode)
        self.palette_arm = arm
        self.encoder_digests = []
        if arm == "original":
            return
        translated = [transplant(model, mean, std) for model in encoders]
        self.encoder_digests = [encoder_digest(m) for m in encoders]
        with torch.no_grad():
            for slot in range(6):
                self.theta[slot, :ENCODER_PARAMETERS].copy_(
                    torch.from_numpy(translated[slot // 2]["theta"])
                )

    def export(self, slot, warm, tokens):
        model = super().export(slot, warm, tokens)
        if self.palette_arm != "original":
            model["palette_arm"] = np.asarray(self.palette_arm)
            model["palette_encoder_digest"] = np.asarray(self.encoder_digests[slot // 2])
        return model
