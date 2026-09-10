"""Decode the preselected INTEL-TAU subset only after estimator selection lock.

Publisher TIFFs are linear, black corrected and saturation normalized; .wp RGB.
No chart mask is needed: publisher excludes the GT acquisition images.
No CCM, camera identity, target statistics or GT enters image preprocessing.
"""

import argparse
import hashlib
import json
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from download_cc_v2_fresh import FROZEN

from luma_skin_vision.cc.core import experts, unit
from luma_skin_vision.cc.data import sample
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def prepare(root, provenance, output, selection_lock):
    if not selection_lock.is_file():
        raise ValueError("Freeze source-selected estimator configuration before decoding target")
    lock = json.loads(selection_lock.read_text(encoding="utf-8"))
    if lock.get("target_errors_observed") is not False:
        raise ValueError("Expected explicit pre-target estimator selection lock")
    if output.exists():
        raise FileExistsError(output)
    manifests = sorted(provenance.glob("*.field1_128.manifest.json"))
    if {path.name: sha256(path) for path in manifests} != FROZEN:
        raise ValueError("Frozen target selection manifests changed")
    output.mkdir(parents=True)
    start = time.perf_counter()
    images, labels, estimators, rows, source_hashes = [], [], [], [], {}
    for manifest_path in manifests:
        source_hashes[manifest_path.name] = sha256(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["samples"]:
            payloads, hashes = {}, {}
            for kind in ("image", "gt"):
                member = record[kind]
                path = (root / member["path"]).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("Unsafe member path")
                content = path.read_bytes()
                if (
                    len(content) != member["uncompressed_bytes"]
                    or f"{zlib.crc32(content) & 0xFFFFFFFF:08x}" != member["crc32"]
                ):
                    raise ValueError("File integrity differs from frozen ZIP record")
                payloads[kind] = content
                hashes[kind] = hashlib.sha256(content).hexdigest()
            rgb = cv2.imdecode(np.frombuffer(payloads["image"], np.uint8), cv2.IMREAD_UNCHANGED)
            if rgb is None or rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("Expected publisher uint16 three-channel linear TIFF")
            rgb = rgb[..., ::-1]
            gt = np.fromstring(payloads["gt"].decode("utf-8-sig"), sep=" ")
            if gt.shape != (3,) or not np.isfinite(gt).all() or not (gt > 0).all():
                raise ValueError(
                    "Expected three finite positive normalized RGB whitepoint components"
                )
            x = rgb.astype(np.float32) / 65535
            x[x.max(axis=-1) >= 0.98] = 0
            images.append(sample(x, 128))
            labels.append(unit(gt))
            estimators.append(experts(x))
            relative = record["image"]["path"]
            rows.append(
                {
                    "id": relative,
                    "camera": relative.split("/")[0],
                    "subset": "test",
                    "group": relative,
                    "reference_hash": hashes["gt"],
                    "group_limitation": "true scene clusters unavailable; group=imageID; reference_hash permits repeated-reference sensitivity analysis",
                    "shape": list(rgb.shape),
                    "sha256": hashes["image"],
                    "gt_sha256": hashes["gt"],
                    "black": 0,
                    "white": 65535,
                    "selection_sha256": record["selection_sha256"],
                }
            )
            if len(rows) % 32 == 0:
                print(
                    json.dumps({"prepared": len(rows), "seconds": time.perf_counter() - start}),
                    flush=True,
                )
    if (
        len(rows) != 384
        or len({r["id"] for r in rows}) != 384
        or sorted(sum(r["camera"] == c for r in rows) for c in {r["camera"] for r in rows})
        != [128, 128, 128]
    ):
        raise ValueError("Expected exact prespecified384 /128each-camera subset")
    np.savez_compressed(
        output / "fresh.npz",
        images=np.stack(images).astype(np.float16),
        gt=np.stack(labels),
        experts=np.stack(estimators),
    )
    write_json(output / "fresh_manifest.json", rows)
    write_json(
        output / "preparation.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "images": len(rows),
            "source_selection_manifests": source_hashes,
            "estimator_selection_lock_sha256": sha256(selection_lock),
            "script_sha256": sha256(Path(__file__)),
            "cache_sha256": sha256(output / "fresh.npz"),
            "manifest_sha256": sha256(output / "fresh_manifest.json"),
            "seconds": time.perf_counter() - start,
            "reference": "https://researchportal.tuni.fi/en/datasets/intel-tau/",
            "preprocessing": "cv2 uint16 BGR->RGB; /65535; reject max>=0.98; no target rectangle; area128square; per-image95th percentile exposure normalization; clamp0to4; FP16cache. No sRGB gamma, CCM, camera-specific values, target-distribution statistics or GT used.",
            "protocol": "source-trained SimpleCube++ ->128preselected unique-field images each unseen INTEL-TAU camera. Custom subset; not author full benchmark.",
            "rights": "original CC BY-SA4.0, evaluation only here; mirrorMITbadge ignored; pinned mirror CRC/hash verified; byteidentity to original multipart files unverified",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/public/intel_tau_v2"))
    parser.add_argument("--provenance", type=Path, default=Path("docs/data/provenance/cc_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/cc_v2_fresh128"))
    parser.add_argument(
        "--selection-lock",
        type=Path,
        default=Path("docs/benchmarks/cc_v2/estimator_selection_lock.json"),
    )
    args = parser.parse_args()
    prepare(args.root, args.provenance, args.output, args.selection_lock)
