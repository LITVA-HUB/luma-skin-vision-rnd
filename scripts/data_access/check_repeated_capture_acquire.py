#!/usr/bin/env python3
"""Offline cache integrity and failure-path checks; fixtures stay in a temp dir."""
import contextlib
import argparse
import hashlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import repeated_capture_acquire as acquisition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, help="Optional new QA receipt path; never replace existing evidence")
    args = parser.parse_args()
    if args.receipt is not None and args.receipt.exists():
        parser.error("QA receipt already exists; use a new path or omit --receipt")
    primary_before = acquisition.PRIMARY_RECEIPT.read_bytes()
    source_receipt = json.loads(primary_before)
    for row in source_receipt["files"]:
        key = (row["dataset"], row["filename"])
        if key in acquisition.EXPECTED_ARCHIVES:
            assert acquisition.EXPECTED_ARCHIVES[key] == (row["bytes"], row["sha256"])
    result = {"pinned_constants_match_primary_receipt": True}
    root = Path("data/public/repeated_capture_access")
    with patch.object(acquisition.urllib.request, "urlopen", side_effect=AssertionError("Network forbidden")):
        rows = [acquisition.acquire(root, dataset, name, url, cache_only=True)
                for dataset, files in acquisition.SOURCES.items() for name, url in files.items()]
    assert all(r["success"] and r["network_performed"] is False for r in rows)
    result["real_cache_offline_passed_files"] = len(rows)
    url = acquisition.SOURCES["georgia_tech"]["labels_gt.zip"]
    original = (root / "georgia_tech" / "labels_gt.zip").read_bytes()
    corrupt = bytes([original[0] ^ 1]) + original[1:]
    with tempfile.TemporaryDirectory(prefix="luma_acquire_qa_", dir="data/public/repeated_capture_access") as name:
        temp = Path(name)
        corrupt_root = temp / "corrupt_cache"
        target = corrupt_root / "georgia_tech" / "labels_gt.zip"
        target.parent.mkdir(parents=True)
        target.write_bytes(corrupt)
        with patch.object(acquisition.urllib.request, "urlopen", side_effect=AssertionError("Network forbidden")):
            row = acquisition.acquire(corrupt_root, "georgia_tech", "labels_gt.zip", url)
        assert not row["success"] and "SHA256 mismatch" in row["error"]
        assert target.read_bytes() == corrupt
        result["same_size_wrong_hash_cache_rejected_without_replacement"] = True
        with patch.object(acquisition.urllib.request, "urlopen", side_effect=AssertionError("Network forbidden")):
            row = acquisition.acquire(temp / "absent_cache", "georgia_tech", "labels_gt.zip", url, cache_only=True)
        assert not row["success"] and "FileNotFoundError" in row["error"]
        result["absent_cache_fails_without_network"] = True

        class FakeResponse(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/zip"}

        response = FakeResponse(corrupt)
        response.url = url
        downloaded = temp / "bad_download"
        with patch.object(acquisition.urllib.request, "urlopen", return_value=response):
            row = acquisition.acquire(downloaded, "georgia_tech", "labels_gt.zip", url)
        assert not row["success"] and "SHA256 mismatch" in row["error"]
        assert not (downloaded / "georgia_tech" / "labels_gt.zip").exists()
        assert (downloaded / "georgia_tech" / "labels_gt.zip.part").read_bytes() == corrupt
        result["downloaded_wrong_hash_not_promoted"] = True
    with patch("sys.argv", ["repeated_capture_acquire.py"]), patch.object(acquisition.urllib.request, "urlopen", side_effect=AssertionError("Network forbidden")), contextlib.redirect_stderr(io.StringIO()):
        try:
            acquisition.main()
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("Expected existing receipt refusal")
    assert acquisition.PRIMARY_RECEIPT.read_bytes() == primary_before
    result["existing_primary_receipt_refused_and_unchanged"] = True
    result["primary_receipt_sha256"] = hashlib.sha256(primary_before).hexdigest()
    result["network_requests_performed"] = 0
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
