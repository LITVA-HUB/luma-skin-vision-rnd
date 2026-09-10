"""Freeze and acquire only previously unused INTEL-TAU members, with original terms."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import download_cc_v2_fresh as downloader

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "docs/data/provenance/cc_v3"
DATA = ROOT / "data/public/intel_tau_v3"
BASE = "ea490d3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remainder(all_rows, used_rows):
    all_map = {s["image"]["path"]: s for s in all_rows}
    used_map = {s["image"]["path"]: s for s in used_rows}
    if len(all_map) != 256 or len(all_rows) != 256 or len(used_map) != 128 or len(used_rows) != 128:
        raise ValueError("Expected unique256 and used128")
    if not used_map.keys() <= all_map.keys() or any(all_map[k] != v for k, v in used_map.items()):
        raise ValueError("Historical member missing or metadata changed")
    result = [s for s in all_rows if s["image"]["path"] not in used_map]
    if len(result) != 128:
        raise ValueError("Expected exact128-image remainder")
    return result


def original(relative):
    local = ROOT / relative
    committed = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT)
    if local.read_bytes() != committed:
        raise ValueError("Original committed metadata bytes changed: " + relative)
    return json.loads(committed), hashlib.sha256(committed).hexdigest()


def freeze():
    lock_path = PROVENANCE / "selection_lock.json"
    if lock_path.exists():
        raise FileExistsError("Do not overwrite the existing pre-acquisition lock")
    official, official_hash = original("docs/data/provenance/cc_v2/official_v3_verified.json")
    license_urls = {v.get("custom_url") for v in official["access_rights"]["license"]}
    if "https://creativecommons.org/licenses/by-sa/4.0/" not in license_urls:
        raise ValueError("Original BY-SA4 grant missing")
    documents, parents = {}, {}
    total = 0
    for camera in ("Canon_5DSR", "Nikon_D810", "Sony_IMX135_BLCCSC"):
        prefix = f"docs/data/provenance/cc_v2/{camera}_1080p.zip.field1_"
        larger, parents[prefix + "256.manifest.json"] = original(prefix + "256.manifest.json")
        used, parents[prefix + "128.manifest.json"] = original(prefix + "128.manifest.json")
        if larger["source"] != used["source"]:
            raise ValueError("Historical archive identity mismatch")
        rows = remainder(larger["samples"], used["samples"])
        total += sum(s["combined_record_range"]["bytes"] for s in rows)
        documents[camera + ".remainder128.json"] = {
            "status": "FROZEN_METADATA_ONLY_NO_NEW_PIXELS_OR_GT_VALUES_READ",
            "source": larger["source"],
            "samples": rows,
            "selection": "Exact256-minus-used128 from immutableV2 metadata; no new ranking or label-based filtering",
            "rights": "OriginalINTEL-TAU CC-BY-SA4.0; separateV3 research training/evaluation track; no blanket unrestricted proprietary weight-distribution clearance",
            "outer_protocol": "FutureLOCO: whole outer camera excluded from all fitting; oldV1/V2 images never fitted. Roles/grouping must be fixed before training.",
        }
    if total > downloader.PAYLOAD_LIMIT:
        raise ValueError("New download exceeds4.5GB initial cap")
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, value in documents.items():
        path = PROVENANCE / name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
        hashes[name] = digest(path)
    lock = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "previous_goal_turn": "progress; frozenV2 result preserved",
        "new_pixels_or_gt_values_read": False,
        "original_metadata_commit": BASE,
        "original_license_record_sha256": official_hash,
        "original_license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "parent_manifests": parents,
        "new_manifests": hashes,
        "planned_images": 384,
        "planned_transfer_bytes": total,
        "retry_traffic_limit": downloader.TRAFFIC_LIMIT,
        "script_sha256": digest(__file__),
        "downloader_sha256": digest(downloader.__file__),
        "data_root": DATA.relative_to(ROOT).as_posix(),
        "original_archive_byte_identity_verified": False,
        "usage_revision": "V2eval-only history remains unchanged. Only newV3 unused images may enter separately recordedBY-SA research weights; no imported restricted/noncommercial data or weights, no publication.",
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"lock": str(lock_path), "images": 384, "bytes": total}), flush=True)


def acquire(verify_only=False):
    lock = json.loads((PROVENANCE / "selection_lock.json").read_text(encoding="utf-8"))
    if (
        digest(__file__) != lock["script_sha256"]
        or digest(downloader.__file__) != lock["downloader_sha256"]
    ):
        raise ValueError("Acquisition code changed after metadata freeze")
    for relative, sha in lock["parent_manifests"].items():
        if digest(ROOT / relative) != sha:
            raise ValueError("Historical exclusion metadata changed")
    if DATA.resolve() == (ROOT / "data/public/intel_tau_v2").resolve():
        raise ValueError("Refusing original data destination")
    # Reuse the independently reviewed exact206/ZIP/CRC/size/path/retry implementation.
    # Overrides exist only in this process; original code/manifests are never edited.
    downloader.PROVENANCE = PROVENANCE
    downloader.FROZEN = lock["new_manifests"]
    sys.argv = ["pinned_v3_range_bridge", "--data-root", str(DATA), "--workers", "4"]
    if verify_only:
        sys.argv.append("--verify-only")
    downloader.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "acquire", "verify"))
    arguments = parser.parse_args()
    if arguments.action == "freeze":
        freeze()
    else:
        acquire(arguments.action == "verify")
