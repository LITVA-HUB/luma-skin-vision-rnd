"""Canonical-frame correction evidence. Known frame normalization; unverified combination."""
import torch
from cc_v5_model import CorrectionEvidenceNet, _illuminant

from luma_skin_vision.cc.v2 import channel_anchor


class CanonicalEvidenceNet(CorrectionEvidenceNet):
    def __init__(self, mode="transport", frame="sog"):
        super().__init__(mode)
        if frame not in {"none", "sog"}:
            raise ValueError("frame must be none or sog")
        self.frame = frame

    def encode(self, image):
        if self.frame == "none":
            cache = super().encode(image)
            cache["anchor_logchroma"] = torch.zeros_like(cache["point_action"])
            return cache
        if (image.ndim != 4 or image.shape[0] == 0 or image.shape[1] != 3
                or min(image.shape[-2:]) < 32 or not image.is_floating_point()):
            raise ValueError("Expected nonempty floating NCHW RGB, H/W >=32")
        with torch.autocast(device_type=image.device.type, enabled=False):
            raw = image.to(torch.float64 if image.dtype == torch.float64 else torch.float32)
            valid = torch.isfinite(raw).all((1, 2, 3)) & (raw >= 0).all((1, 2, 3))
            clean = torch.nan_to_num(raw, nan=0., posinf=0., neginf=0.).clamp_min(0)
            scale = clean.amax((1, 2, 3), keepdim=True)
            valid &= (clean.amax((-2, -1)) > 0).all(-1)
            scaled = clean / torch.where(scale > 0, scale, torch.ones_like(scale))
            scaled = torch.where(valid[:, None, None, None], scaled, torch.ones_like(scaled))
            anchor = channel_anchor(scaled, p=6)
            canonical = scaled / anchor[:, :, None, None]
            cache = super().encode(canonical)
            cache["valid"] = cache["valid"] & valid
            loganchor = anchor.log()
            cache["anchor_logchroma"] = (loganchor[:, [0, 2]] - loganchor[:, 1:2]).to(cache["point_action"].dtype)
            return cache

    def select(self, cache, steps=2):
        result = super().select(cache, steps)
        # Queries/actions remain residual-frame coordinates. Only final outputs
        # return to camera RGB; the training objective consumes residual GT.
        anchor = cache["anchor_logchroma"]
        result["absolute_trajectory_actions"] = result["trajectory_actions"] + anchor[:, None]
        if self.frame != "none":
            result["pred"] = _illuminant(result["action"] + anchor)
            result["base_pred"] = _illuminant(cache["point_action"] + anchor)
        return result
