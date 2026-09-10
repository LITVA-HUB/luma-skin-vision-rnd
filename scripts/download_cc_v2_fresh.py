"""Acquire the frozen INTEL-TAU 3 x 128 field-1 subset by verified ZIP ranges.

Checks only bytes, sizes, CRCs and hashes; does not decode TIFFs or read GT values.
Data license is upstream CC BY-SA 4.0, irrespective of the mirror's MIT badge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "docs/data/provenance/cc_v2"
REVISION = "9cd6308d3cb2e383d6b185abd16fa1958bf6bc54"
FROZEN = {
    "Canon_5DSR_1080p.zip.field1_128.manifest.json": "0f6674ae6a27fe65cd027b7031905a8023705e4bd1bf390cc27a6dafbbc3d710",
    "Nikon_D810_1080p.zip.field1_128.manifest.json": "e0ff94721e79dd48edfe2949cdb7cb4c649024cddff5bc1c1ccfc80a1c873972",
    "Sony_IMX135_BLCCSC_1080p.zip.field1_128.manifest.json": "06488e11251c6f56ac21f45fbec931b990e7c1b10e80b8a22c65c47778b3fccc",
}
PAYLOAD_LIMIT = 4_500_000_000
TRAFFIC_LIMIT = 6_000_000_000


def safe_destination(root: Path, name: str) -> Path:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe member path")
    result = (root / Path(*rel.parts)).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Member escaped data root")
    return result


def byte_record(data: bytes, member: dict) -> dict:
    crc = f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"
    if len(data) != member["uncompressed_bytes"] or crc != member["crc32"]:
        raise ValueError("Member CRC or uncompressed length mismatch")
    return {
        "path": member["path"],
        "bytes": len(data),
        "crc32": crc,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def unpack_member(blob: bytes, origin: int, member: dict) -> bytes:
    offset = member["local_header_offset"] - origin
    if offset < 0 or offset + 30 > len(blob):
        raise ValueError("Member local header outside range")
    fields = struct.unpack_from("<4s5H3I2H", blob, offset)
    signature, _, flags, method, _, _, crc, packed, unpacked, name_len, extra_len = fields
    if signature != b"PK\x03\x04" or flags != member["flags"] or method != member["compression"]:
        raise ValueError("Local and central ZIP headers disagree")
    if flags & (1 | 8) or method not in (0, 8):
        raise ValueError("Encrypted or descriptor-based ZIP members unsupported")
    if (
        crc != int(member["crc32"], 16)
        or packed != member["compressed_bytes"]
        or unpacked != member["uncompressed_bytes"]
    ):
        raise ValueError("Local and central member sizes/CRC disagree")
    begin = offset + 30
    name = blob[begin : begin + name_len].decode("utf-8" if flags & 2048 else "cp437")
    if name != member["path"]:
        raise ValueError("Local ZIP filename mismatch")
    begin += name_len + extra_len
    end = begin + packed
    if end > len(blob) or unpacked > 50_000_000:
        raise ValueError("Member size cap exceeded")
    if method == 0:
        return blob[begin:end]
    decoder = zlib.decompressobj(-15)
    data = decoder.decompress(blob[begin:end], unpacked + 1)
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(data) != unpacked:
        raise ValueError("Invalid deflate stream")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/public/intel_tau_v2")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, digest in FROZEN.items():
        raw = (PROVENANCE / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Frozen manifest changed: {name}")
        manifest = json.loads(raw)
        source = manifest["source"]
        if source["revision"] != REVISION or len(manifest["samples"]) != 128:
            raise ValueError("Unexpected source or subset count")
        expected_url = f"https://huggingface.co/datasets/presencesw/INTEL-TAU/resolve/{REVISION}/{source['archive']}"
        if source["source_url"] != expected_url:
            raise ValueError("Unexpected acquisition URL")
        jobs.extend((source, sample) for sample in manifest["samples"])
    successful_budget = sum(x[1]["combined_record_range"]["bytes"] for x in jobs)
    if successful_budget > PAYLOAD_LIMIT:
        raise ValueError("Successful acquisition would exceed 4.5 GB")
    journal = data_root / "range_transfer_journal.jsonl"
    previous = (
        [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    )
    reserved = sum(x["reserved_bytes"] for x in previous if x["event"] == "reserve")
    lock = threading.Lock()
    started = time.monotonic()
    records, failures = [], []
    progress = {"last": started}

    def log(event: dict) -> None:
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()

    def acquire(source: dict, sample: dict) -> dict:
        nonlocal reserved
        members = [sample["image"], sample["gt"]]
        paths = [safe_destination(data_root, m["path"]) for m in members]
        # Rehash and verify central-directory CRC before skipping existing bytes.
        if all(p.is_file() for p in paths):
            try:
                checked = [byte_record(p.read_bytes(), m) for p, m in zip(paths, members)]
                return {
                    "archive": source["archive"],
                    "image": sample["image"]["path"],
                    "state": "verified_existing",
                    "files": checked,
                }
            except ValueError:
                if args.verify_only:
                    raise
        if args.verify_only:
            raise ValueError("Missing or invalid files in verify-only mode")
        bounds = sample["combined_record_range"]
        begin, end, length = bounds["start"], bounds["end"], bounds["bytes"]
        if end - begin + 1 != length or length > 50_000_000:
            raise ValueError("Invalid planned byte range")
        url = source["source_url"] + f"?cc_v2_range={begin}-{end}"
        expected_range = f"bytes {begin}-{end}/{source['archive_bytes']}"
        for attempt in range(1, 4):
            with lock:
                if reserved + length > TRAFFIC_LIMIT:
                    raise ValueError("Cumulative reserved transfer budget exceeds 6 GB")
                reserved += length
                log(
                    {
                        "event": "reserve",
                        "image": sample["image"]["path"],
                        "reserved_bytes": length,
                        "attempt": attempt,
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
            got = 0
            try:
                request = Request(
                    url,
                    headers={
                        "Range": f"bytes={begin}-{end}",
                        "Accept-Encoding": "identity",
                        "User-Agent": "Luma-CC-v2/1",
                    },
                )
                with urlopen(request, timeout=45) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected_range
                    ):
                        raise ValueError("Server did not honor the exact requested Range")
                    declared = response.headers.get("Content-Length")
                    if declared is not None and int(declared) != length:
                        raise ValueError("Content-Length differs from planned range")
                    chunks = []
                    while got < length:
                        chunk = response.read(min(1_048_576, length - got))
                        if not chunk:
                            raise ValueError("Truncated HTTP range")
                        chunks.append(chunk)
                        got += len(chunk)
                blob = b"".join(chunks)
                payloads = [unpack_member(blob, begin, m) for m in members]
                checked = [byte_record(b, m) for b, m in zip(payloads, members)]
                for path, payload in zip(paths, payloads):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = path.with_name(path.name + ".partial")
                    partial.write_bytes(payload)
                    partial.replace(path)
                with lock:
                    log(
                        {
                            "event": "complete",
                            "image": sample["image"]["path"],
                            "body_bytes": got,
                            "status": 206,
                            "content_range": expected_range,
                            "range_sha256": hashlib.sha256(blob).hexdigest(),
                        }
                    )
                return {
                    "archive": source["archive"],
                    "image": sample["image"]["path"],
                    "state": "downloaded",
                    "http_status": 206,
                    "content_range": expected_range,
                    "range_bytes": got,
                    "files": checked,
                }
            except Exception as exc:
                with lock:
                    log(
                        {
                            "event": "failed",
                            "image": sample["image"]["path"],
                            "body_bytes": got,
                            "error": f"{type(exc).__name__}: {str(exc).split('?')[0]}",
                        }
                    )
                if attempt == 3:
                    raise
                time.sleep(attempt)
        raise AssertionError("Unreachable")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(acquire, source, sample): sample for source, sample in jobs}
        for future in as_completed(pending):
            sample = pending[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "image": sample["image"]["path"],
                        "error": f"{type(exc).__name__}: {str(exc).split('?')[0]}",
                    }
                )
            now = time.monotonic()
            if now - progress["last"] > 20 or len(records) + len(failures) == len(jobs):
                print(
                    json.dumps(
                        {
                            "complete_images": len(records),
                            "failed_images": len(failures),
                            "planned_images": len(jobs),
                            "reserved_bytes": reserved,
                            "elapsed_seconds": round(now - started, 1),
                        }
                    ),
                    flush=True,
                )
                progress["last"] = now
    journal_events = (
        [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    )
    report = {
        "status": "VERIFIED_BYTES_ONLY" if not failures and len(records) == 384 else "INCOMPLETE",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "license": "CC-BY-SA-4.0 (original INTEL-TAU data); mirror MIT label not adopted",
        "source_revision": REVISION,
        "original_archive_byte_identity_verified": False,
        "provenance_limitation": "Official multipart sizes match mirror totals; whole official part SHA256s cannot verify sparse mirror ranges. CRCs validate membership/integrity, not publisher authenticity.",
        "images_or_gt_values_inspected": False,
        "frozen_manifests": FROZEN,
        "planned_images": len(jobs),
        "verified_images": len(records),
        "verified_files": sum(len(r["files"]) for r in records),
        "planned_successful_range_bytes": successful_budget,
        "cumulative_reserved_transfer_bytes": reserved,
        "cumulative_response_body_bytes": sum(x.get("body_bytes", 0) for x in journal_events),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "records": sorted(records, key=lambda x: x["image"]),
        "failures": failures,
    }
    name = "verification_report.json" if args.verify_only else "download_report.json"
    (PROVENANCE / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"records", "frozen_manifests"}}),
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
