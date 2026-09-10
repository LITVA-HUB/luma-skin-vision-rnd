import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

VERSION = "srgb-exif-v1"


def decode_image(path, *, mirrored=False):
    if Path(path).stat().st_size > 30 * 1024 * 1024:
        raise ValueError("image file exceeds 30 MiB limit")
    with Image.open(path) as source:
        if source.width * source.height > 24_000_000:
            raise ValueError("image exceeds 24 megapixels")
        image = ImageOps.exif_transpose(source)
        if mirrored:
            image = ImageOps.mirror(image)
        icc = image.info.get("icc_profile")
        if icc:
            try:
                image = ImageCms.profileToProfile(
                    image,
                    ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                    ImageCms.createProfile("sRGB"),
                    outputMode="RGB",
                )
            except Exception as exc:
                raise ValueError("ICC conversion failed; color interpretation unsupported") from exc
        elif image.mode not in ("RGB", "RGBA", "L"):
            raise ValueError("Unprofiled non-RGB image is unsupported")
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def quality_gate(rgb, bbox):
    """Provisional acquisition thresholds, NOT calibrated error/quality probabilities."""
    issues = []
    x, y, w, h = bbox
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > rgb.shape[1] or y + h > rgb.shape[0]:
        return ["invalid_face_geometry"]
    if min(w, h) < 96:
        issues.append("face_too_small")
    face = rgb[int(y) : int(y + h), int(x) : int(x + w)]
    if not np.isfinite(face).all():
        return issues + ["invalid_pixels"]
    clipped = np.mean((face.min(axis=-1) <= 0.01) | (face.max(axis=-1) >= 0.99))
    if clipped > 0.15:
        issues.append("excessive_clipping")
    lum = face.mean(axis=-1)
    lap = -4 * lum[1:-1, 1:-1] + lum[:-2, 1:-1] + lum[2:, 1:-1] + lum[1:-1, :-2] + lum[1:-1, 2:]
    if lap.size == 0 or lap.var() < 1e-5:
        issues.append("blur_or_low_texture")
    return issues


def region_tensor(rgb, mask, resolution):
    yy, xx = np.where(mask)
    if len(xx) == 0:
        raise ValueError("empty region")
    # Keep encoded RGB signal; mask non-ROI pixels to zero. No ImageNet normalization.
    crop = rgb[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1].copy()
    crop *= mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1, None]
    image = Image.fromarray(np.uint8(np.round(np.clip(crop, 0, 1) * 255)))
    resized = (
        np.asarray(
            image.resize((resolution, resolution), Image.Resampling.BILINEAR), dtype=np.float32
        )
        / 255
    )
    return resized.transpose(2, 0, 1).copy()
