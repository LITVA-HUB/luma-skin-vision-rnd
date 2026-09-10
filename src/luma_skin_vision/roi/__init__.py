"""Coarse cheek geometry and deterministic measurement suitability proxies.

Bounding-box cheeks are provisional. They are not learned skin segmentation.
No skin-color threshold is used to exclude darker skin by assumption.
"""

import numpy as np

from luma_skin_vision.color import delta_e00, srgb_to_lab
from luma_skin_vision.photometry import hypotheses


def cheek_masks(shape, bbox):
    height, width = shape[:2]
    x, y, w, h = bbox
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError("bbox outside image")
    yy, xx = np.mgrid[:height, :width]
    return [
        ((xx - (x + cx * w)) / (0.115 * w)) ** 2 + ((yy - (y + 0.64 * h)) / (0.11 * h)) ** 2 <= 1
        for cx in (0.72, 0.28)
    ]


def usable_mask(rgb):
    return np.isfinite(rgb).all(axis=-1) & (rgb.min(axis=-1) > 0.01) & (rgb.max(axis=-1) < 0.99)


def robust_rgb(rgb, masks, trim=0.1):
    pixels = np.asarray(rgb)[np.logical_or.reduce(masks) & usable_mask(rgb)]
    if len(pixels) < 8:
        raise ValueError("insufficient usable ROI pixels")
    pixels = np.sort(pixels, axis=0)
    cut = int(len(pixels) * trim)
    return pixels[cut : len(pixels) - cut].mean(axis=0)


def measure(rgb, masks):
    return srgb_to_lab(robust_rgb(rgb, masks))


def ambiguity_features(rgb, masks):
    candidates = np.array([measure(h, masks) for h in hypotheses(rgb).values()])
    regions = np.array([measure(rgb, [m]) for m in masks])
    pooled = np.logical_or.reduce(masks)
    pix = rgb[pooled]
    perturb = [measure(rgb, [np.roll(m, shift, axis=1) for m in masks]) for shift in (-1, 1)]
    lum = rgb.mean(axis=-1)
    features = [
        *candidates.std(axis=0),
        float(delta_e00(regions[0], regions[-1])),
        float(np.mean(delta_e00(candidates, candidates[0]))),
        float(np.mean(delta_e00(perturb, candidates[0]))),
        float((~usable_mask(rgb)[pooled]).mean()),
        float(np.mean(pix.max(axis=1) > 0.95)),
        float(np.mean(pix.mean(axis=1) < 0.05)),
        float(np.abs(np.diff(lum, axis=0)).mean()),
        float(np.abs(np.diff(lum, axis=1)).mean()),
        float(pix.std(axis=0).mean()),
    ]
    return np.asarray(features, dtype=np.float32)
