#!/usr/bin/env python3
"""Reuse the official STW Faces95 release for repeated-identity qualification.

No network, new labels, skin parser, learned model, or source-image publication.
"""
import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from repeated_capture_qualify import stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stw-root", type=Path, default=Path("data/public/stw_access"))
    p.add_argument("--private-out", type=Path, default=Path("data/public/repeated_capture_access/stw_faces95"))
    p.add_argument("--summary", type=Path, default=Path("docs/data/access_2026_09_15/repeated_capture/stw_faces95_qualification.json"))
    a = p.parse_args()
    with (a.stw_root / "all_annotated_data.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == "Faces 95"]
    assert len(rows) == 1440
    a.private_out.mkdir(exist_ok=True, parents=True)
    records = []
    dimensions, modes = collections.Counter(), collections.Counter()
    exif_count = icc_count = color_count = 0
    with zipfile.ZipFile(a.stw_root / "images.zip") as z:
        names = set(z.namelist())
        for r in rows:
            member = "images/all_cropped_images_by_class/" + r["class"] + "/" + r["new_tokens"] + PurePosixPath(r["paths"]).stem + "_face_1.jpg"
            assert member in names
            blob = z.read(member)
            person = "F95P_" + hashlib.sha256(("LUMA_STW_F95|" + r["tokens"]).encode()).hexdigest()[:12]
            capture = int(PurePosixPath(r["paths"]).stem.rsplit(".", 1)[1])
            with Image.open(io.BytesIO(blob)) as im:
                im.load()
                dimensions[f"{im.width}x{im.height}"] += 1
                modes[im.mode] += 1
                exif_count += bool(im.getexif())
                icc_count += bool(im.info.get("icc_profile"))
                rgb = np.asarray(im.convert("RGB"))
                color_count += bool(np.any(rgb[..., 0] != rgb[..., 1]) or np.any(rgb[..., 1] != rgb[..., 2]))
                y0, y1, x0, x1 = im.height // 4, 3 * im.height // 4, im.width // 4, 3 * im.width // 4
                region = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64) / 255
                median = np.median(region, axis=0)
                records.append({"person": person, "image": person + f"_C{capture:02}", "capture_index": capture,
                                "zip_member": member, "original_source_path": r["paths"],
                                "sha256": hashlib.sha256(blob).hexdigest(),
                                "width": im.width, "height": im.height,
                                "median_encoded_rgb": median.tolist(),
                                "encoded_brightness_proxy": float(median @ np.array([.2126, .7152, .0722])),
                                "camera": None, "session": None, "illumination_id": None,
                                "source_apparent_mst_class": int(r["class"]), "instrument_lab": None})
    records.sort(key=lambda r: r["image"])
    groups = collections.defaultdict(list)
    for r in records:
        groups[r["person"]].append(r)
    assert len(groups) == 72 and all(len(g) == 20 and {r["capture_index"] for r in g} == set(range(1, 21)) for g in groups.values())
    pair_rgb, pair_brightness, pair_records = [], [], []
    for person, rows in sorted(groups.items()):
        for x, y in itertools.combinations(rows, 2):
            d = float(np.linalg.norm(np.asarray(x["median_encoded_rgb"]) - y["median_encoded_rgb"]))
            b = abs(x["encoded_brightness_proxy"] - y["encoded_brightness_proxy"])
            pair_rgb.append(d)
            pair_brightness.append(b)
            pair_records.append({"person": person, "image_a": x["image"], "image_b": y["image"],
                                 "relation": "same_source_identity; real distinct captures; exact illumination difference unknown"})
    manifest = a.private_out / "manifest.private.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    (a.private_out / "pairs.private.json").write_text(json.dumps(pair_records, indent=2) + "\n")
    result = {"dataset": "STW official Faces95 derived full-face crops", "downloaded": True,
              "people": len(groups), "images": len(records), "decoded": len(records), "decode_failures": 0,
              "repeated_captures_per_person": 20, "within_person_pairs": len(pair_records),
              "distinct_file_sha256": len({r["sha256"] for r in records}),
              "dimensions": dict(dimensions), "image_modes": dict(modes), "true_color_images": color_count,
              "exif_images": exif_count, "icc_images": icc_count,
              "camera_metadata_images": 0, "session_metadata_images": 0, "illumination_metadata_images": 0,
              "pair_relation": "same source token identity; capture number 1..20; no lighting target assigned",
              "source_transformation": "Official STW MediaPipe face crops resized to 300x300, JPEG, not original Essex frames",
              "within_person_median_encoded_rgb_euclidean": stats(pair_rgb),
              "within_person_encoded_brightness_proxy_difference": stats(pair_brightness),
              "diagnostic_roi": "central 150x150 of derived face crop; includes non-skin; not absolute color or illumination GT",
              "usable_for": "repeated identity/capture invariance research candidate; upstream rights unverified; no Lab target",
              "rights_status": "STW instructs following each source license; Essex source page returns 502/timeouts; exact research grant not retrieved",
              "original_essex_bytes_obtained": False,
              "downloaded_official_stw_derivative": True,
              "additional_network_bytes": 0,
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    a.summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
