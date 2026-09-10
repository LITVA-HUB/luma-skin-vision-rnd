# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MODIFIED/ADAPTED 2026: independent PyTorch translation of formulas from
# google/ffcc commit 2fa9e1316954dbd3913630b7d597927941b4dd32.
# Original source/license retained in docs/research/cc_v3_sources/google_ffcc/.
# Source files: ChannelizeImage.m, MaskedLocalAbsoluteDeviation.m, Pad1.m,
# FeaturizeImage.m, Psplat2.m, EvaluateModel.m, FitBivariateVonMises.m,
# UvToP.m, UvToIdx.m, RgbToUv.m, UvToRgb.m, PrecomputeTrainingData.m,
# TrainModel.m, TrainModelLossfun.m, projects/DefaultHyperparams.m and
# projects/GehlerShi/GehlerShiConstants.m (internal/ except where noted).
"""FFCC-inspired, source-grounded compact control; not benchmark reproduction.

Only floating linear RGB with explicit [0,1] full scale is accepted. No resize,
black-level subtraction, quantization, chart/saturation detection, data loading,
optimizer or pretrained weights is included. Real spatial parameters preserve
the full-FFT quadratic objective but differ from packed/preconditioned minFunc
optimization. Zero circular resultants are explicitly undefined, with finite
diagnostic decoding and confidence0; train cross entropy before Gaussian NLL.

Batch-first API: RGB Nx3xHxW; mask NxHxW; hist Nx2xnxn; PMF Nxnxn.
All histograms and toroidal moments are computed in FP64. The model casts
histograms to parameter dtype for FFT filtering; model.double() gives FP64
filtering too. Nearest histogram inputs are nondifferentiable; filters/bias,
moment decode and objectives are differentiable away from chart switches.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

SOURCE_COMMIT = "2fa9e1316954dbd3913630b7d597927941b4dd32"


def round_matlab(x):
    """Round nearest with half ties away from zero, as MATLAB round."""
    return x.sign() * torch.floor(x.abs() + .5)


def _validate_grid(n, h, lo):
    if n < 2 or n % 2 or h <= 0 or not math.isfinite(h) or not math.isfinite(lo):
        raise ValueError("Expected even n>=2, finite h>0 and finite lo")


def rgb_to_uv(rgb):
    """Positive RGB Nx3 -> (log G-log R, log G-log B); exposure cancels."""
    if rgb.ndim != 2 or rgb.shape[-1] != 3 or not torch.isfinite(rgb).all() or not (rgb > 0).all():
        raise ValueError("Expected positive finite RGB Nx3")
    logged = rgb.double().log()
    return torch.stack([logged[:, 1] - logged[:, 0], logged[:, 1] - logged[:, 2]], -1)


def uv_to_rgb(uv):
    """Projective illuminant decode; remove common log scale before exponent."""
    log_rgb = torch.stack([-uv[:, 0], torch.zeros_like(uv[:, 0]), -uv[:, 1]], -1)
    rgb = (log_rgb - log_rgb.amax(-1, keepdim=True)).exp()
    return F.normalize(rgb, dim=-1)


def periodic_histogram(uv, valid, n=64, h=1/32, lo=-.4375):
    """Nearest periodic unit-count histogram: uv NxPx2; u rows, v columns.

    Each histogram normalizes separately. Empty histograms stay exactly zero;
    returned count is the number of included pixels, before normalization.
    """
    _validate_grid(n, h, lo)
    if uv.ndim != 3 or uv.shape[-1] != 2 or valid.shape != uv.shape[:2]:
        raise ValueError("Expected uv NxPx2 and valid NxP")
    valid = valid.bool() & torch.isfinite(uv).all(-1)
    safe = torch.where(valid[..., None], uv.double(), torch.zeros_like(uv, dtype=torch.float64))
    indices = round_matlab((safe - lo) / h).remainder(n).long()
    flat = indices[..., 0] * n + indices[..., 1]
    hist = torch.zeros(uv.shape[0], n*n, dtype=torch.float64, device=uv.device)
    hist = hist.scatter_add(1, flat, valid.double())
    count = valid.sum(-1)
    hist = hist / count.double().clamp_min(torch.finfo(torch.float64).eps)[:, None]
    return hist.reshape(-1, n, n), count


def masked_local_absolute_deviation(rgb, mask):
    """Source double path: eight masked neighbors, replicated image/mask edges.

    No valid neighbors yields NaN, which featurize explicitly excludes.
    No integer casting/bitshift path is implemented.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3 or mask.shape != (rgb.shape[0], *rgb.shape[-2:]):
        raise ValueError("Expected RGB Nx3xHxW and mask NxHxW")
    rgb = rgb.double()
    weights = mask[:, None].double()
    padded = F.pad(rgb, (1, 1, 1, 1), mode="replicate")
    padded_weights = F.pad(weights, (1, 1, 1, 1), mode="replicate")
    height, width = rgb.shape[-2:]
    numerator = torch.zeros_like(rgb)
    denominator = torch.zeros_like(weights)
    for row in range(3):
        for col in range(3):
            if row == col == 1:
                continue
            neighbor = padded[:, :, row:row+height, col:col+width]
            weight = weights * padded_weights[:, :, row:row+height, col:col+width]
            numerator = numerator + weight * (neighbor - rgb).abs()
            denominator = denominator + weight
    result = numerator / denominator.clamp_min(1)
    return torch.where(denominator > 0, result, torch.full_like(result, float("nan")))


def featurize(rgb, mask=None, n=64, h=1/32, lo=-.4375, min_feature=1/256):
    """Two independently normalized histograms and source all-pixel linear mean.

    Float full scale1 resolves the source's ambiguous isa(im,'float') branch.
    Default mask follows GehlerShi MASK_ZERO_PIXELS. An explicit mask affects
    histograms only: average_rgb uses every original spatial pixel, including
    existing zero-filled pixels, exactly as PrecomputeTrainingData.m specifies.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point() or rgb.numel() == 0:
        raise ValueError("Expected nonempty floating RGB Nx3xHxW")
    if not torch.isfinite(rgb).all():
        raise ValueError("RGB must be finite")
    if not ((rgb >= 0) & (rgb <= 1)).all():
        raise ValueError("Floating linear RGB must be in [0, 1]")
    if min_feature <= 0:
        raise ValueError("min_feature must be positive")
    if mask is None:
        mask = (rgb > 0).all(1)
    if mask.dtype != torch.bool or mask.shape != (rgb.shape[0], *rgb.shape[-2:]):
        raise ValueError("mask must be boolean NxHxW")
    rgb64 = rgb.double()
    channels = (rgb64 * mask[:, None], masked_local_absolute_deviation(rgb64, mask))
    histograms, counts = [], []
    for feature in channels:
        valid = mask & torch.isfinite(feature).all(1) & (feature >= min_feature).all(1)
        safe = torch.where(valid[:, None], feature, torch.ones_like(feature))
        logged = safe.log()
        uv = torch.stack([logged[:, 1] - logged[:, 0], logged[:, 1] - logged[:, 2]], -1)
        hist, count = periodic_histogram(uv.reshape(rgb.shape[0], -1, 2), valid.flatten(1), n, h, lo)
        histograms.append(hist)
        counts.append(count)
    average = rgb64.mean((-2, -1))
    average_valid = (average > 0).all(-1)
    average = F.normalize(average, dim=-1)
    return {"hist": torch.stack(histograms, 1), "average_rgb": average,
            "average_valid": average_valid, "feature_counts": torch.stack(counts, 1)}


def circular_score(hist, filters, bias):
    """MATLAB fft2/ifft2 convention: no conjugation, shift or gain map."""
    if hist.ndim != 4 or hist.shape[1:] != filters.shape or bias.shape != hist.shape[-2:]:
        raise ValueError("Expected hist Nx2xnxn, filters2xnxn, bias nxn")
    frequency = (torch.fft.fft2(hist) * torch.fft.fft2(filters)[None]).sum(1)
    logits = torch.fft.ifft2(frequency).real + bias
    pmf = logits.flatten(1).softmax(-1).reshape_as(logits)
    return logits, pmf


def decode(pmf, lo=-.4375, h=1/32, eps_bins=1., average_rgb=None, unwrap_mode="gray_light"):
    """Circular mean and source rounded-chart covariance, with pad only.

    Mean validity requires both circular resultant lengths>1e-10. Undefined
    axes get a finite zero-index diagnostic; confidence is then0. This explicit
    refusal is an adaptation, not a confident source estimate for uniform PMFs.
    Output covariance and means use FP64; no extra covariance jitter is added.
    """
    n = pmf.shape[-1]
    _validate_grid(n, h, lo)
    if pmf.ndim != 3 or pmf.shape[-2] != n or eps_bins <= 0:
        raise ValueError("Expected PMF Nxnxn and eps_bins>0")
    if unwrap_mode not in {"gray_light", "gray_world"}:
        raise ValueError("unwrap_mode must be gray_light or gray_world")
    if not torch.isfinite(pmf).all() or not (pmf >= 0).all() or not torch.allclose(pmf.sum((-2, -1)), torch.ones_like(pmf[:, 0, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("PMF must be finite nonnegative and normalized")
    p = pmf.double()
    p = p / p.sum((-2, -1), keepdim=True)
    marginals = torch.stack([p.sum(-1), p.sum(-2)], 1)
    bins0 = torch.arange(n, dtype=p.dtype, device=p.device)
    theta = bins0 * (2 * math.pi / n)
    sine = (marginals * theta.sin()).sum(-1)
    cosine = (marginals * theta.cos()).sum(-1)
    resultant = torch.sqrt(sine.square() + cosine.square())
    axis_valid = resultant > 1e-10
    angle = torch.atan2(torch.where(axis_valid, sine, torch.zeros_like(sine)),
                        torch.where(axis_valid, cosine, torch.ones_like(cosine)))
    mu_index = angle.remainder(2 * math.pi) * (n / (2 * math.pi))
    # Use the original one-based chart exactly; rounding/chart selection has
    # zero derivative away from its discontinuities.
    shifted = ((bins0 + 1)[None, None] - round_matlab(mu_index + 1)[:, :, None] + n/2 - 1).remainder(n) + 1
    expectation = (marginals * shifted).sum(-1)
    variance = (marginals * shifted.square()).sum(-1) - expectation.square()
    cross = (p * shifted[:, 0, :, None] * shifted[:, 1, None, :]).sum((-2, -1)) - expectation[:, 0] * expectation[:, 1]
    covariance = torch.stack([variance[:, 0], cross, cross, variance[:, 1]], -1).reshape(-1, 2, 2)
    covariance = (covariance + eps_bins * torch.eye(2, dtype=p.dtype, device=p.device)) * h*h
    mu_uv = lo + h * mu_index
    anchor_valid = torch.ones(p.shape[0], dtype=torch.bool, device=p.device)
    if unwrap_mode == "gray_world":
        if average_rgb is None or average_rgb.shape != (p.shape[0], 3):
            raise ValueError("gray_world requires average_rgb Nx3")
        anchor_valid = torch.isfinite(average_rgb).all(-1) & (average_rgb > 0).all(-1)
        safe_average = torch.where(anchor_valid[:, None], average_rgb, torch.ones_like(average_rgb))
        anchor_uv = rgb_to_uv(safe_average)
        period = n * h
        mu_uv = mu_uv - period * round_matlab((mu_uv - anchor_uv) / period)
    mean_valid = axis_valid.all(-1) & anchor_valid
    confidence_raw = torch.exp(math.log(eps_bins * h*h) - .5 * torch.linalg.slogdet(covariance).logabsdet)
    confidence = torch.where(mean_valid, confidence_raw, torch.zeros_like(confidence_raw))
    pred = uv_to_rgb(mu_uv)
    return {"mu_index": mu_index, "mu_uv": mu_uv, "covariance_uv": covariance,
            "pred": pred, "resultant": resultant, "mean_valid": mean_valid,
            "confidence_raw": confidence_raw, "confidence": confidence,
            "anchor_valid": anchor_valid}


def target_distribution(gt_rgb, n=64, h=1/32, lo=-.4375, kind="nearest"):
    """Source SMOOTH_CROSS_ENTROPY=true means nearest; false means bilinear."""
    _validate_grid(n, h, lo)
    uv = rgb_to_uv(gt_rgb)
    if kind == "nearest":
        return periodic_histogram(uv[:, None], torch.ones(uv.shape[0], 1, dtype=torch.bool, device=uv.device), n, h, lo)[0]
    if kind != "bilinear":
        raise ValueError("target kind must be nearest or bilinear")
    position = (uv - lo) / h
    low = position.floor()
    fraction = position - low
    target = torch.zeros(uv.shape[0], n*n, dtype=uv.dtype, device=uv.device)
    for u_offset in range(2):
        for v_offset in range(2):
            u = (low[:, 0].long() + u_offset).remainder(n)
            v = (low[:, 1].long() + v_offset).remainder(n)
            weight = (fraction[:, 0] if u_offset else 1 - fraction[:, 0]) * (fraction[:, 1] if v_offset else 1 - fraction[:, 1])
            target = target.scatter_add(1, (u*n + v)[:, None], weight[:, None])
    return target.reshape(-1, n, n)


def _reduce(values, reduction):
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be none, mean or sum")


def cross_entropy(logits, gt_rgb, h=1/32, lo=-.4375, kind="nearest", reduction="mean"):
    target = target_distribution(gt_rgb, logits.shape[-1], h, lo, kind).to(logits.dtype)
    log_p = logits.flatten(1).log_softmax(-1)
    return _reduce(-(target.flatten(1) * log_p).sum(-1), reduction)


def gaussian_uv_nll(output, gt_rgb, h=1/32, eps_bins=1., reduction="mean"):
    """Shifted Gaussian UV likelihood from source; residual is NOT wrapped.

    Requires defined decoded means for every row. Undefined circular means
    require CE warmup or an explicit caller policy; no zero-loss masking occurs.
    This is not the exact BVM density and is not an RGB angular training loss.
    """
    if not output["mean_valid"].all():
        raise ValueError("Gaussian UV likelihood requires defined circular means for every row")
    delta = rgb_to_uv(gt_rgb) - output["mu_uv"]
    covariance = output["covariance_uv"]
    cholesky = torch.linalg.cholesky(covariance)
    solved = torch.cholesky_solve(delta[..., None], cholesky).squeeze(-1)
    half_logdet = cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    loss = .5 * (delta * solved).sum(-1) + half_logdet + math.log(2 * math.pi) - math.log(2 * math.pi * eps_bins * h*h)
    return _reduce(loss, reduction)


def fft_regularizer(filters, bias, data_mass=1., filter_lambda=None, filter_shift=(2**-8, 2**-8), bias_lambda=1., bias_shift=2**-8):
    """Full-FFT quadratic source objective, including MATLAB/Parseval scaling.

    For averaged data loss use data_mass1; for a sum use the sum of sample
    weights. This is not L1 TV and is not interchangeable with AdamW decay.
    """
    n = filters.shape[-1]
    if filters.shape != (2, n, n) or bias.shape != (n, n) or data_mass <= 0:
        raise ValueError("Expected two square filters, matching bias and data_mass>0")
    kernel = filters.new_zeros(2, n, n)
    kernel[:, 0, 0] = -1 / math.sqrt(8)
    kernel[0, 1, 0] = kernel[1, 0, 1] = 1 / math.sqrt(8)
    variation = torch.fft.fft2(kernel).abs().square().sum(0)
    if filter_lambda is None:
        filter_lambda = (n**-4, n**-4)
    lam = torch.as_tensor(filter_lambda, dtype=filters.dtype, device=filters.device).reshape(2, 1, 1)
    shift = torch.as_tensor(filter_shift, dtype=filters.dtype, device=filters.device).reshape(2, 1, 1)
    filter_weights = lam * variation + shift
    bias_weights = bias_lambda * variation + bias_shift
    total = (filter_weights * torch.fft.fft2(filters).abs().square()).sum()
    total = total + (bias_weights * torch.fft.fft2(bias).abs().square()).sum()
    return .5 * data_mass * total


class FFCCInspiredNet(nn.Module):
    """Two spatial-domain filters plus bias, default12,288 real parameters."""

    def __init__(self, n=64, h=1/32, lo=-.4375, eps_bins=1., unwrap_mode="gray_light"):
        super().__init__()
        _validate_grid(n, h, lo)
        self.n, self.h, self.lo = n, h, lo
        self.eps_bins, self.unwrap_mode = eps_bins, unwrap_mode
        self.filters = nn.Parameter(torch.zeros(2, n, n))
        self.bias = nn.Parameter(torch.zeros(n, n))

    def forward_histograms(self, hist, average_rgb=None):
        hist = hist.to(self.filters.dtype)
        logits, pmf = circular_score(hist, self.filters, self.bias)
        output = decode(pmf, self.lo, self.h, self.eps_bins, average_rgb, self.unwrap_mode)
        feature_valid = (hist.sum((-2, -1)) > 0).any(-1)
        output["valid"] = feature_valid & output["mean_valid"] & (output["pred"] > 0).all(-1)
        return {"hist": hist, "logits": logits, "pmf": pmf, "feature_valid": feature_valid, **output}

    def forward(self, rgb, mask=None):
        features = featurize(rgb, mask, self.n, self.h, self.lo)
        output = self.forward_histograms(features["hist"], features["average_rgb"])
        return {**features, **output}

    def regularizer(self, data_mass=1., **kwargs):
        return fft_regularizer(self.filters, self.bias, data_mass=data_mass, **kwargs)
