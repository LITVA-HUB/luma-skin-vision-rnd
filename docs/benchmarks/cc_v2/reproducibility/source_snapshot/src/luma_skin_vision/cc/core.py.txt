import hashlib

import cv2
import numpy as np

EXPERT_NAMES = ["gray_world", "max_rgb", "shades_gray", "gray_edge"]


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm <= 0) or not np.isfinite(x).all():
        raise ValueError("Nonfinite or zero color vector")
    return x / norm


def angular(pred, gt):
    return np.degrees(np.arccos(np.clip(np.sum(unit(pred) * unit(gt), axis=-1), -1, 1)))


def reproduction(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    if np.any(pred <= 0) or np.any(gt <= 0):
        raise ValueError("Reproduction requires positive illuminants")
    corrected = gt / pred
    return angular(corrected, np.ones_like(corrected))


def summarize(errors):
    e = np.sort(np.asarray(errors, dtype=float))
    if e.size == 0 or not np.isfinite(e).all() or np.any(e < 0):
        raise ValueError("Invalid errors")
    q1, median, q3 = np.percentile(e, [25, 50, 75])
    tail = max(1, int(np.ceil(len(e) / 4)))
    return {
        "n": len(e),
        "mean": float(e.mean()),
        "median": float(median),
        "trimean": float((q1 + 2 * median + q3) / 4),
        "best25": float(e[:tail].mean()),
        "worst25": float(e[-tail:].mean()),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "max": float(e[-1]),
        "over10_fraction": float(np.mean(e > 10)),
        "over20_fraction": float(np.mean(e > 20)),
    }


def selective_curve(errors, scores, ids):
    e, s = np.asarray(errors), np.asarray(scores)
    if len(e) != len(s) or len(e) != len(ids) or not np.isfinite(s).all():
        raise ValueError("Invalid selective inputs")
    order = np.array(
        sorted(range(len(e)), key=lambda i: (s[i], hashlib.sha256(ids[i].encode()).hexdigest()))
    )
    ranked = e[order]
    means = np.cumsum(ranked) / np.arange(1, len(e) + 1)
    fixed = {}
    for percent in [100, 95, 90, 80, 70, 60]:
        n = max(1, int(np.floor(len(e) * percent / 100)))
        fixed[str(percent)] = {**summarize(ranked[:n]), "coverage": n / len(e)}
    return {
        "fixed": fixed,
        "coverage": (np.arange(1, len(e) + 1) / len(e)).tolist(),
        "risk": means.tolist(),
        "aurc": float(means.mean()),
        "order": order.tolist(),
    }


def linearize(rgb, black=2048, white=15000):
    if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected 16-bit linear RGB PNG")
    if white <= black:
        raise ValueError("Invalid sensor levels")
    x = np.clip((rgb.astype(np.float32) - black) / (white - black), 0, 1)
    saturated = np.max(rgb, axis=2) >= white
    x[saturated] = 0
    return x


def experts(rgb):
    # Input already linear camera RGB, with invalid/target pixels zeroed.
    valid = (rgb.max(axis=2) > 1e-5) & (rgb.max(axis=2) < 1)
    pixels = rgb[valid].astype(np.float64)
    if len(pixels) == 0:
        return np.tile(unit([1, 1, 1]), (4, 1))
    gw = pixels.mean(axis=0)
    mx = pixels.max(axis=0)
    sog = np.mean(pixels**6, axis=0) ** (1 / 6)
    smooth = cv2.GaussianBlur(rgb, (0, 0), 1)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    # Exclude a 5-pixel neighborhood of masked pixels and image boundaries.
    inner = cv2.erode(
        valid.astype(np.uint8),
        np.ones((11, 11), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    edges = np.sqrt(dx**2 + dy**2)[inner].astype(np.float64)
    ge = np.mean(edges**6, axis=0) ** (1 / 6) if len(edges) else gw
    if not np.any(ge > 1e-10):
        ge = gw
    return unit(np.maximum(np.stack([gw, mx, sog, ge]), 1e-8))
