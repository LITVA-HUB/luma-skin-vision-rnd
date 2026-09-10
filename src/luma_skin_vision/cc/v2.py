"""Matched direct and anchor-relative color-constancy models, without novelty claims.

For positive diagonal gains D and anchors above EPS, a(Dx)=D a(x).
The anchored encoder consequently sees the same x/a(x), and its illuminant
transforms equivariantly, up to L2 normalization. This algebra is not evidence
that real camera/ISP changes are diagonal, nor a guarantee of physical accuracy.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

EPS = 1e-8
CHANNELS = ("r", "g", "b")
CHEAP_FEATURE_COLUMNS = (
    [
        f"log_pred_over_{anchor}_{channel}_minus_green"
        for anchor in ("gw", "sog", "max")
        for channel in ("r", "b")
    ]
    + [
        f"patch2x2_{row}{col}_{channel}_over_global_gw"
        for channel in CHANNELS
        for row in range(2)
        for col in range(2)
    ]
    + [f"spatial_std_{channel}_over_global_gw" for channel in CHANNELS]
)
FEATURE_UNITS = "dimensionless; natural log ratios for first six cheap columns"


def validate_image(x):
    """No input clipping: reject unsupported pixels before computing an anchor."""
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("Expected a floating point tensor")
    if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[0], x.shape[2], x.shape[3]) < 1:
        raise ValueError("Expected nonempty NCHW RGB images")
    torch._assert(torch.isfinite(x).all(), "Image values must be finite")
    torch._assert((x >= 0).all(), "Image values must be nonnegative")


def _anchor(x, p):
    if p == 1:
        return x.mean(dim=(-2, -1)).clamp_min(EPS)
    # Homogeneous scaling avoids overflow from directly taking the sixth power.
    maximum = x.amax(dim=(-2, -1)).clamp_min(EPS)
    scaled = x / maximum[:, :, None, None]
    power = scaled.pow(p).mean(dim=(-2, -1)).clamp_min(torch.finfo(x.dtype).tiny)
    return (maximum * power.pow(1 / p)).clamp_min(EPS)


def channel_anchor(x, p=1):
    """Unnormalized channelwise Minkowski mean, including masked zero pixels.

    Returns Nx3. The positive numerical floor breaks exact homogeneity only near
    degenerate channels. Such samples are flagged invalid by input_validity().
    """
    validate_image(x)
    if p not in (1, 6):
        raise ValueError("Supported anchor powers are 1 and 6")
    return _anchor(x, p)


def input_validity(x):
    """Each channel must have a mean above EPS; this is not a quality certificate."""
    validate_image(x)
    return (x.mean(dim=(-2, -1)) > EPS).all(dim=-1)


class CompactResidualCC(nn.Module):
    def __init__(self, mode="direct", backbone="small"):
        super().__init__()
        if mode not in ("direct", "gw", "sog") or backbone not in ("small", "large"):
            raise ValueError("Unsupported mode or backbone")
        self.mode = mode
        self.backbone_name = backbone
        factory, width = (
            (mobilenet_v3_small, 576) if backbone == "small" else (mobilenet_v3_large, 960)
        )
        self.backbone = factory(weights=None).features
        self.context = nn.Sequential(nn.Linear(width, 64), nn.SiLU())
        self.illuminant = nn.Linear(64, 3)
        nn.init.zeros_(self.illuminant.weight)
        nn.init.zeros_(self.illuminant.bias)

    def forward(self, image):
        validate_image(image)
        # Branches depend only on immutable architecture, never on pixel values.
        if self.mode == "direct":
            rms = image.square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(EPS**2).sqrt()
            encoder_input = image / rms
            anchor = torch.ones_like(image[:, :, 0, 0])
        else:
            anchor = _anchor(image, 1 if self.mode == "gw" else 6)
            encoder_input = torch.log1p((image / anchor[:, :, None, None]).clamp(max=8))
        context = self.context(self.backbone(encoder_input).mean(dim=(-2, -1)))
        residual = self.illuminant(context).clamp(-2, 2).exp()
        return F.normalize(anchor * residual, dim=-1), context


def gain_augment(x, gt, magnitude=0.7, *, exposure_magnitude=0.5, generator=None):
    """Transformed-real augmentation under a diagonal model, not new physical GT.

    No post-gain clipping is applied. Metadata makes the exact transformation
    auditable; zero magnitude still applies common exposure and random flips.
    """
    validate_image(x)
    if gt.shape != (len(x), 3) or not gt.is_floating_point() or gt.device != x.device:
        raise ValueError("Expected same-device Nx3 floating ground truth")
    torch._assert((torch.isfinite(gt) & (gt > 0)).all(), "Ground truth must be finite positive")
    if (
        not math.isfinite(magnitude)
        or not 0 <= magnitude <= 2
        or not math.isfinite(exposure_magnitude)
        or exposure_magnitude < 0
    ):
        raise ValueError("Invalid augmentation magnitudes")
    options = {"device": x.device, "dtype": x.dtype, "generator": generator}
    gains = ((torch.rand((len(x), 3), **options) * 2 - 1) * magnitude).exp()
    exposure = ((torch.rand((len(x),), **options) * 2 - 1) * exposure_magnitude).exp()
    flipped = torch.rand((len(x),), **options) < 0.5
    augmented = x * gains[:, :, None, None] * exposure[:, None, None, None]
    augmented = torch.where(flipped[:, None, None, None], augmented.flip(-1), augmented)
    target = F.normalize(gt * gains, dim=-1)
    return augmented, target, {"gains": gains, "exposure": exposure, "flipped": flipped}


def risk_features_invariant(x, pred, context):
    """Relative thumbnail features; no absolute RGB/prediction or camera metadata.

    Cheap columns are invariant when pred transforms equivariantly and channels
    stay above the floor. Context is invariant only for the anchored models.
    The direct mode has no diagonal-invariance promise. Invalid rows are finite
    fallbacks and must be rejected regardless of any learned risk score.
    """
    validate_image(x)
    if pred.shape != (len(x), 3) or context.shape != (len(x), 64):
        raise ValueError("Expected Nx3 prediction and Nx64 context")
    torch._assert((torch.isfinite(pred) & (pred > 0)).all(), "Prediction must be finite positive")
    torch._assert(torch.isfinite(context).all(), "Context must be finite")
    gw, sog = _anchor(x, 1), _anchor(x, 6)
    maximum = x.amax(dim=(-2, -1)).clamp_min(EPS)
    ratios = []
    for anchor in (gw, sog, maximum):
        log_ratio = pred.log() - anchor.log()
        ratios.append(log_ratio[:, [0, 2]] - log_ratio[:, 1:2])
    normalized = x / gw[:, :, None, None]
    patches = F.adaptive_avg_pool2d(normalized, (2, 2)).flatten(1)
    spatial_std = normalized.flatten(2).std(dim=-1, correction=0)
    cheap = torch.cat([*ratios, patches, spatial_std], dim=1)
    return {
        "context": context,
        "cheap": cheap,
        "combined": torch.cat([context, cheap], dim=1),
        "valid": input_validity(x),
    }
