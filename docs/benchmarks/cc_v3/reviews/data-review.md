# CC V3 data selection and acquisition review

Review date: 2026-09-10  
Scope: local, metadata-only review of `scripts/cc_v3_data.py`, `tests/test_cc_v3_data.py`, `docs/data/cc_v3_usage_decision.md`, `docs/data/provenance/cc_v3/selection_lock.json`, the six frozen V2 256/128 manifests, and `scripts/download_cc_v2_fresh.py`. No network acquisition was run and no TIFF or ground-truth value was decoded or inspected.

## Decision

**PASS — no blocking finding for the authorized bounded acquisition.**

The three V3 manifests contain exactly 384 previously unused image/GT pairs: 128 for each of Canon 5DSR, Nikon D810, and Sony IMX135 BLCCSC. For every camera, the new ordered sample list is exactly the immutable V2 256-item metadata list minus its immutable used 128-item list. The old and new populations have zero image-ID overlap, including across the combined three-camera population, and all retained sample metadata is byte-derived from the corresponding 256-item parent entry.

## Independent checks

- All six parent-manifest SHA-256 values match `selection_lock.json` and the bytes at milestone commit `ea490d3` used by `original()`.
- All three remainder manifests contain 128 unique image paths. Their SHA-256 values match the lock: Canon `75dbf454...7200ef`, Nikon `57bc51e4...04c30`, Sony `ab834962...f2dc1`.
- The derived transfer totals are Canon 1,498,461,789 bytes, Nikon 1,457,452,094 bytes, and Sony 1,480,494,468 bytes, totaling exactly **4,436,408,351 bytes**. This is below the downloader's decimal 4.5 GB successful-payload cap; its cumulative reservation/retry ceiling is **6,000,000,000 bytes**.
- Each selected combined record range has `end - start + 1 == bytes`, is nonnegative, stays below the declared pinned archive length, and contains the selected image and GT local-header offsets.
- Each manifest source is pinned to Hugging Face dataset revision `9cd6308d3cb2e383d6b185abd16fa1958bf6bc54`, the expected camera ZIP name, and an exact HTTPS resolve URL. The frozen downloader reconstructs and compares that URL, requires HTTP 206 with the exact `Content-Range`, bounds each range at 50 MB, verifies local ZIP header fields, decompresses only supported members, then verifies uncompressed size and CRC before writing.
- The current downloader is byte-identical to `scripts/download_cc_v2_fresh.py` at commit `ea490d3`; SHA-256 `998fa1b4...5de740` matches the V3 lock.
- A local stubbed bridge probe confirmed that `acquire()` changes only the downloader's in-process provenance root and frozen-manifest map, then supplies explicit argv with `--data-root .../data/public/intel_tau_v3 --workers 4`. The adapter rejects the V2 destination. The reused downloader consequently places its transfer journal and extracted members under the V3 data root and its report under V3 provenance; no V2 data/provenance write path is used by this invocation.
- The original publisher metadata at `ea490d3` hashes to `7a4a6758...79bf`, matching the lock, and records `https://creativecommons.org/licenses/by-sa/4.0/`. The usage decision preserves attribution/ShareAlike duties, distinguishes historical V2 evaluation-only use, and explicitly states that BY-SA lineage does not automatically clear unrestricted proprietary model-weight redistribution.
- `python -m pytest tests/test_cc_v3_data.py -q` passed: **4 passed**. The tests cover exact ordered subtraction and reject duplicate, missing, or metadata-changed historical entries.

## Residual limitations

This review does not establish byte identity between the pinned mirror ZIPs and the publisher's original multipart archives. The lock and usage decision correctly record `original_archive_byte_identity_verified: false`; the downloader verifies sparse range structure/CRC and the pinned mirror, not publisher authenticity. Live server range behavior and downloaded payload integrity remain to be established by the authorized acquisition itself.

The checked-in unit tests exercise selection logic only. The adapter hash/destination/argv bridge was verified here with an in-process stub rather than a committed test; this is a non-blocking coverage gap, not an observed protocol defect.
