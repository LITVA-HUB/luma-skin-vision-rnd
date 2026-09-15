#!/usr/bin/env python3
"""Decode official repeated-face files and save privacy-safe inventory statistics.

All geometry is existing author face boxes, not skin labels. RGB diagnostics
are encoded-pixel statistics, never Lab, reflectance, or an illumination GT.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import itertools
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "min": float(x.min()), "max": float(x.max())}


def qualify(root: Path, name: str):
    is_gt = name == "georgia_tech"
    archive = root / name / ("gt_db.zip" if is_gt else "att_faces.zip")
    boxes = {}
    if is_gt:
        with zipfile.ZipFile(root / name / "labels_gt.zip") as z:
            for member in z.namelist():
                fields = z.read(member).decode().split()
                number = int(Path(member).name[3:])
                subject = (number - 1) // 20 + 1
                capture = (number - 1) % 20 + 1
                if len(fields) != 5 or int(fields[-1][1:]) != subject or capture > 15:
                    raise ValueError("Label-to-image mapping failed")
                boxes[(subject, capture)] = [int(v) for v in fields[:4]]
    records = []
    modes = collections.Counter()
    sizes = collections.Counter()
    parsed_color_pixels = 0
    iccs = exifs = 0
    camera_models = collections.Counter()
    exif_color_spaces = collections.Counter()
    dates = set()
    out = root / name / "images"
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        pattern = r"gt_db/s(\d+)/(\d+)\.jpg" if is_gt else r"s(\d+)/(\d+)\.pgm"
        members = [(m, re.fullmatch(pattern, m)) for m in z.namelist()]
        for member, match in members:
            if match is None:
                continue
            subject, capture = map(int, match.groups())
            raw = z.read(member)
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                modes[im.mode] += 1
                sizes[f"{im.width}x{im.height}"] += 1
                iccs += bool(im.info.get("icc_profile"))
                exif = im.getexif()
                exifs += bool(exif)
                exif_ifd = exif.get_ifd(34665) if exif.get(34665) else {}
                make = str(exif.get(271, "")).rstrip("\0 ")
                model = str(exif.get(272, "")).rstrip("\0 ")
                camera_model = (make + " " + model).strip() or None
                timestamp = exif_ifd.get(36867)
                date = timestamp[:10] if timestamp and re.fullmatch(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", timestamp) else None
                if date:
                    dates.add(date)
                if camera_model:
                    camera_models[camera_model] += 1
                color_space = exif_ifd.get(40961)
                if color_space is not None:
                    exif_color_spaces[str(color_space)] += 1
                rgb = np.asarray(im.convert("RGB"))
                is_color = bool(np.any(rgb[..., 0] != rgb[..., 1]) or np.any(rgb[..., 1] != rgb[..., 2]))
                parsed_color_pixels += is_color
                bbox = boxes[(subject, capture)] if is_gt else [0, 0, im.width, im.height]
                x0, y0, x1, y1 = bbox
                if not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height):
                    raise ValueError("Author face box outside decoded image")
                # Central half of face bounding rectangle; no skin or instrument claim.
                cx0, cy0 = int(x0 + .25 * (x1 - x0)), int(y0 + .25 * (y1 - y0))
                cx1, cy1 = int(x0 + .75 * (x1 - x0)), int(y0 + .75 * (y1 - y0))
                region = rgb[cy0:cy1, cx0:cx1].reshape(-1, 3).astype(np.float64) / 255
                median = np.median(region, axis=0)
                suffix = ".jpg" if is_gt else ".pgm"
                pseudonym = f"P{subject:03d}_C{capture:02d}"
                path = out / (pseudonym + suffix)
                if path.exists() and path.read_bytes() != raw:
                    raise ValueError("Existing image differs; refuse overwrite")
                if not path.exists():
                    path.write_bytes(raw)
                records.append({"image": pseudonym, "person": f"P{subject:03d}", "capture_index": capture,
                                "original_member": member, "path": str(path),
                                "sha256": hashlib.sha256(raw).hexdigest(), "width": im.width, "height": im.height,
                                "face_bbox_xyxy": bbox, "diagnostic_box_xyxy": [cx0, cy0, cx1, cy1],
                                "median_encoded_rgb": median.tolist(),
                                "encoded_brightness_proxy": float(median @ np.array([.2126, .7152, .0722])),
                                "clipping_fraction": float(np.mean(np.any((region <= 1 / 255) | (region >= 254 / 255), axis=1))),
                                "camera_model": camera_model, "camera_unit_id": None,
                                "capture_datetime_original": timestamp, "capture_date_proxy": date,
                                "exif_color_space": color_space,
                                "session": None, "illumination": None, "instrument_lab": None})
    records.sort(key=lambda r: r["image"])
    private_manifest = root / name / "manifest.private.json"
    private_manifest.write_text(json.dumps(records, indent=2) + "\n")
    groups = collections.defaultdict(list)
    for row in records:
        groups[row["person"]].append(row)
    all_distances = []
    all_brightness_changes = []
    same_date_brightness_changes = []
    different_date_brightness_changes = []
    per_person = []
    for person, rows in sorted(groups.items()):
        distance = []
        brightness = []
        for a, b in itertools.combinations(rows, 2):
            distance.append(float(np.linalg.norm(np.asarray(a["median_encoded_rgb"]) - b["median_encoded_rgb"])))
            brightness.append(abs(a["encoded_brightness_proxy"] - b["encoded_brightness_proxy"]))
            if a["capture_date_proxy"] and b["capture_date_proxy"]:
                collector = same_date_brightness_changes if a["capture_date_proxy"] == b["capture_date_proxy"] else different_date_brightness_changes
                collector.append(brightness[-1])
        all_distances.extend(distance)
        all_brightness_changes.extend(brightness)
        per_person.append({"person": person, "images": len(rows), "pairs": len(distance),
                           "known_capture_dates": len(set(r["capture_date_proxy"] for r in rows if r["capture_date_proxy"])),
                           "within_person_median_encoded_rgb_euclidean": stats(distance),
                           "within_person_brightness_proxy_difference": stats(brightness)})
    summary = {"dataset": name, "downloaded": True, "people": len(groups), "images": len(records),
               "decoded": len(records), "decode_failures": 0, "images_per_person": dict(collections.Counter(len(x) for x in groups.values())),
               "same_person_pairs": len(all_distances), "source_image_modes": dict(modes), "dimensions": dict(sizes),
               "true_nonidentical_rgb_channels_images": parsed_color_pixels,
               "identical_rgb_channels_images": len(records) - parsed_color_pixels,
               "icc_images": iccs, "exif_images": exifs,
               "unique_file_sha256": len(set(r["sha256"] for r in records)),
               "author_face_boxes": len(boxes), "validated_author_face_boxes": len(records) if is_gt else 0,
               "camera_models": dict(camera_models), "camera_model_metadata_images": sum(camera_models.values()),
               "physical_camera_unit_ids_present": 0, "session_ids_present": 0, "illumination_ids_present": 0,
               "capture_date_metadata_images": sum(r["capture_date_proxy"] is not None for r in records),
               "unique_capture_dates": len(dates),
               "people_by_known_capture_date_count": dict(collections.Counter(r["known_capture_dates"] for r in per_person)),
               "exif_color_spaces": dict(exif_color_spaces),
               "capture_index_is_not_session_or_illumination": True,
               "native_color_space": "749 JPEGs self-declare sRGB by EXIF ColorSpace=1; one unspecified; no ICC or instrument calibration" if is_gt else "8-bit grayscale PGM",
               "target": "same-person membership only", "absolute_lab_target": False,
               "within_person_median_encoded_rgb_euclidean": stats(all_distances),
               "within_person_brightness_proxy_difference": stats(all_brightness_changes),
               "within_person_same_date_brightness_proxy_difference": stats(same_date_brightness_changes) if same_date_brightness_changes else None,
               "within_person_different_date_brightness_proxy_difference": stats(different_date_brightness_changes) if different_date_brightness_changes else None,
               "variability_caveat": "Central face rectangle includes non-skin; observed changes mix light, pose, expression, alignment and capture. These statistics are not an illumination ground truth.",
               "private_manifest_sha256": hashlib.sha256(private_manifest.read_bytes()).hexdigest(),
               "per_person": per_person}
    if is_gt:
        summary["rights_status"] = "RIGHTS_UNRESOLVED: author public face-research download, but no explicit dataset use grant or license found in page/README/archive; not cleared for training or redistribution"
    else:
        summary["rights_status"] = "Author archive supplies download with attribution instruction to AT&T Laboratories Cambridge / Olivetti Research Laboratory; no standardized license or commercial grant asserted"
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/public/repeated_capture_access"))
    p.add_argument("--out", type=Path, default=Path("docs/data/access_2026_09_15/repeated_capture/qualification.json"))
    a = p.parse_args()
    summaries = [qualify(a.data_root, name) for name in ("georgia_tech", "att_orl")]
    a.out.write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in s.items() if k != "per_person"} for s in summaries], indent=2))


if __name__ == "__main__":
    main()
