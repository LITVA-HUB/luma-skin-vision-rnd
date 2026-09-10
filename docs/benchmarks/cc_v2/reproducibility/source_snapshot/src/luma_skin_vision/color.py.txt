"""sRGB (IEC transfer) <-> CIELAB, D65 white / CIE 1931 2 degree observer.

CIEDE2000 follows Sharma et al., DOI 10.1002/col.20070, with kL=kC=kH=1.
These conversions describe encoded colors, not inferred physical reflectance.
"""

import numpy as np

WHITE = np.array([0.95047, 1.0, 1.08883])
RGB_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def _triples(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError("Expected finite color triples")
    return a


def srgb_to_linear(rgb):
    rgb = _triples(rgb)
    if np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("sRGB must be in [0,1]")
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = _triples(rgb)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)


def srgb_to_lab(rgb):
    xyz = srgb_to_linear(rgb) @ RGB_XYZ.T / WHITE
    d = 6 / 29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3 * d**2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def lab_to_srgb(lab, *, clip=True):
    lab = _triples(lab)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    d = 6 / 29
    xyz = np.where(f > d, f**3, 3 * d**2 * (f - 4 / 29)) * WHITE
    rgb = linear_to_srgb(xyz @ np.linalg.inv(RGB_XYZ).T)
    return np.clip(rgb, 0, 1) if clip else rgb


def delta_e00(lab1, lab2):
    x, y = np.broadcast_arrays(_triples(lab1), _triples(lab2))
    l1, a1, b1 = np.moveaxis(x, -1, 0)
    l2, a2, b2 = np.moveaxis(y, -1, 0)
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    cm = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(cm**7 / (cm**7 + 25**7)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = np.degrees(np.arctan2(b1, ap1)) % 360
    hp2 = np.degrees(np.arctan2(b2, ap2)) % 360
    zero = cp1 * cp2 == 0
    dh = hp2 - hp1
    dh = np.where(zero, 0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dl, dc = l2 - l1, cp2 - cp1
    d_h = 2 * np.sqrt(cp1 * cp2) * np.sin(np.radians(dh / 2))
    lm, cpm = (l1 + l2) / 2, (cp1 + cp2) / 2
    hsum = hp1 + hp2
    hm = np.where(
        zero,
        hsum,
        np.where(
            np.abs(hp1 - hp2) <= 180,
            hsum / 2,
            np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
        ),
    )
    t = (
        1
        - 0.17 * np.cos(np.radians(hm - 30))
        + 0.24 * np.cos(np.radians(2 * hm))
        + 0.32 * np.cos(np.radians(3 * hm + 6))
        - 0.20 * np.cos(np.radians(4 * hm - 63))
    )
    sl = 1 + 0.015 * (lm - 50) ** 2 / np.sqrt(20 + (lm - 50) ** 2)
    sc, sh = 1 + 0.045 * cpm, 1 + 0.015 * cpm * t
    rt = (
        -2
        * np.sqrt(cpm**7 / (cpm**7 + 25**7))
        * np.sin(np.radians(60 * np.exp(-(((hm - 275) / 25) ** 2))))
    )
    return np.sqrt(
        np.maximum(0, (dl / sl) ** 2 + (dc / sc) ** 2 + (d_h / sh) ** 2 + rt * dc / sc * d_h / sh)
    )
