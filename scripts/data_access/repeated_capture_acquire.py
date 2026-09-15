#!/usr/bin/env python3
"""Acquire only the official, explicitly enumerated repeated-face releases.

No training, mirrors, login, agreement acceptance, or participant publication.
Existing complete downloads are reused; SHA256 and ZIP CRCs are rechecked.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "georgia_tech": {
        "page.html": "https://www.anefian.com/research/face_reco.htm",
        "README.txt": "https://www.anefian.com/research/GTDB_README.txt",
        "gt_db.zip": "https://www.anefian.com/research/gt_db.zip",
        "labels_gt.zip": "https://www.anefian.com/research/labels_gt.zip",
    },
    "att_orl": {
        "page.html": "https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html",
        "att_faces.zip": "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip",
    },
}

# Pins from the immutable first successful acquisition receipt, 2026-09-15.
# CRC alone only proves internal archive consistency, not source identity.
EXPECTED_ARCHIVES = {
    ("georgia_tech", "gt_db.zip"): (133192489, "2c4e379ef7c3cc5580eb409673a1e6eb75c5e5a7efd9bf3beb1de8059e835cbf"),
    ("georgia_tech", "labels_gt.zip"): (91007, "7f964e398516c26c75491e2c8512620e90a0a1291b7dec2797657a32d7da0850"),
    ("att_orl", "att_faces.zip"): (3769022, "509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec"),
}
PRIMARY_RECEIPT = Path("docs/data/access_2026_09_15/repeated_capture/acquisition.json")


def validate_file(path: Path, dataset: str, filename: str) -> dict:
    expected = EXPECTED_ARCHIVES.get((dataset, filename))
    size = path.stat().st_size
    if expected and size != expected[0]:
        raise ValueError(f"Pinned archive size mismatch: expected {expected[0]}, found {size}")
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if expected and digest != expected[1]:
        raise ValueError(f"Pinned archive SHA256 mismatch: expected {expected[1]}, found {digest}")
    result = {"bytes": size, "sha256": digest, "pinned_archive_verified": expected is not None}
    if filename.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                raise ValueError("Archive CRC validation failed")
            result.update(zip_members=len(z.infolist()), zip_crc_checked=True)
    return result


def acquire(root: Path, dataset: str, filename: str, url: str, cache_only: bool = False) -> dict:
    path = root / dataset / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"dataset": dataset, "filename": filename, "requested_url": url}
    started = time.monotonic()
    try:
        if path.exists():
            receipt["cache_reused"] = True
            receipt["network_performed"] = False
            receipt.update(validate_file(path, dataset, filename))
        else:
            if cache_only:
                raise FileNotFoundError("Requested cache-only verification but source file is absent")
            part = path.with_suffix(path.suffix + ".part")
            size = 0
            # Exclusive create preserves any partial acquisition for investigation.
            with part.open("xb") as out, urllib.request.urlopen(url, timeout=40) as response:
                receipt.update(final_url=response.url, status=response.status,
                               content_type=response.headers.get("Content-Type"), network_performed=True)
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > 250_000_000:
                        raise ValueError("Per-file acquisition cap exceeded")
                    out.write(block)
            receipt.update(validate_file(part, dataset, filename))
            # Link fails if another process created path; never replace that file.
            os.link(part, path)
            part.unlink()
            receipt["cache_reused"] = False
        receipt["success"] = True
    except Exception as error:
        receipt.update(success=False, error=f"{type(error).__name__}: {error}")
    receipt["seconds"] = time.monotonic() - started
    return receipt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/public/repeated_capture_access"))
    p.add_argument("--receipt", type=Path, help="New receipt path; an existing receipt is never replaced")
    p.add_argument("--verify-cache-only", action="store_true", help="Forbid network; print verification unless a NEW --receipt is supplied")
    args = p.parse_args()
    receipt_path = args.receipt if args.receipt is not None else (None if args.verify_cache_only else PRIMARY_RECEIPT)
    if receipt_path is not None and receipt_path.exists():
        p.error("Receipt already exists; preserve it and use --verify-cache-only or an explicit NEW --receipt path")
    jobs = [(args.data_root, dataset, name, url, args.verify_cache_only) for dataset, items in SOURCES.items() for name, url in items.items()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda a: acquire(*a), jobs))
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive mode also closes the race after the early exists check.
        with receipt_path.open("x") as f:
            json.dump({"date": "2026-09-15", "cache_only": args.verify_cache_only, "files": receipts}, f, indent=2)
            f.write("\n")
    print(json.dumps(receipts, indent=2))
    if not all(r["success"] for r in receipts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
