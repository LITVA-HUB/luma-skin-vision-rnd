# INTEL-TAU fresh-camera acquisition

Audit date: 2026-09-10. **Measured metadata and bounded acquisition**, separate from model evaluation. Selection was frozen before any candidate image or white-point value was inspected. Training, selection and calibration remain SimpleCube++ only.

## Decision and provenance

Acquired the deterministic **128 camera-unique field scenes per camera** candidate: 384 images, with their paired `.wp` files. Canon 5DSR and Nikon D810 provide two cameras/sensors absent from SimpleCube++ training and the earlier C5Sony30 pilot. The Sony selection excludes field IDs 001–030. This is a prespecified field-scene subset, not the full INTEL-TAU benchmark or a guarantee of scene/person independence.

The [original Metax V3 dataset record](https://metax.fairdata.fi/v3/datasets/f0570a3f-3d77-4f44-9ef1-99ab4878f17c) is now reachable using PowerShell's normal system TLS trust. Python's default CA bundle rejected the same host; certificate validation was not disabled. Its license field explicitly supplies **CC BY-SA 4.0**, matching [Research.fi](https://research.fi/en/results/dataset/f0570a3f-3d77-4f44-9ef1-99ab4878f17c). Saved record: [official_v3_verified.json](provenance/cc_v2/official_v3_verified.json). It identifies version 4, original URN `urn:nbn:fi:att:f8b62270-d471-4036-b427-f21bce32b965`, dataset modification `2021-04-27T13:14:33Z`, repository record modification `2025-02-10T16:59:14Z`.

The mirror is `presencesw/INTEL-TAU` at immutable Git revision **`9cd6308d3cb2e383d6b185abd16fa1958bf6bc54`**. Its MIT badge is not adopted as a data license. Original attribution: F. Laakom, J. Raitoharju, J. Nikkanen, A. Iosifidis and M. Gabbouj, *INTEL-TAU: A Color Constancy Dataset*, IEEE Access 9, 39560–39567 (2021). Preserve upstream BY-SA terms for images and applicable derived data. No mirror-authored code or pretrained weights are adopted.

Original metadata lists multipart ZIPs whose aggregate lengths exactly equal the mirror archives. However, official SHA-256 checksums apply to 6–8 GB parts; they cannot authenticate individual sparse ranges or be combined into a whole-archive SHA-256. **Original byte identity remains unverified.** ZIP CRCs and local file hashes establish consistency with the pinned mirror, not publisher authenticity. The matching archive names, aggregate lengths and 7,022 paired scene entries strengthen lineage evidence but do not replace original-byte verification.

## Original host and byte ranges

Read-only metadata endpoints succeeded with HTTP 200:

```text
https://metax.fairdata.fi/v3/datasets/f0570a3f-3d77-4f44-9ef1-99ab4878f17c
https://metax.fairdata.fi/v3/datasets/f0570a3f-3d77-4f44-9ef1-99ab4878f17c/files
https://metax.fairdata.fi/v3/datasets/f0570a3f-3d77-4f44-9ef1-99ab4878f17c/files?limit=20&offset=20
```

All 32 original file records, byte sizes, storage identifiers and SHA-256 checksums are saved in [page 1](provenance/cc_v2/official_v3_files_verified.json) and [page 2](provenance/cc_v2/official_v3_files_page2_verified.json). The actual files use `Nikon_D810`, resolving the older descriptive `D180` typo.

The deployed Etsin frontend reveals `POST https://etsin.fairdata.fi/api/v3/download/authorize` with JSON `{"cr_id":"f0570a3f-3d77-4f44-9ef1-99ab4878f17c","file":"/Nikon_D810_1080p.zip.004"}`. This returns a short-lived authorized URL on `download.fairdata.fi`; no account was created. Requested `Range: bytes=6415857098-6415857119` on that original last part returned **HTTP 200, application/octet-stream, no Content-Range, no Accept-Ranges, no Content-Length**. The stream was closed without consuming its body. Therefore original-host sparse extraction was not verified. [Response evidence](provenance/cc_v2/official_nikon_eocd_retry.bin.request.json). The alternative Metax `/v3/download/authorize` route returned 405, `Download is not enabled`; this is distinct from the working Etsin authorization endpoint.

All three pinned mirror URLs honor exact **HTTP 206** requests, including ZIP64 footer and central-directory ranges. The observed final response host is `us.aws.cdn.hf.co`. The successful metadata-only ZIP inspection consumed **2,049,241 response-body bytes**, including the pinned Hugging Face tree listing. Ancillary original-record and frontend-route discovery stayed well below the separate 20 MB metadata bound. No image or GT payload was consumed during that phase.

| Camera archive / exact pinned URL | Archive bytes | Central-directory range, inclusive | Directory bytes |
|---|---:|---|---:|
| [Canon_5DSR_1080p.zip](https://huggingface.co/datasets/presencesw/INTEL-TAU/resolve/9cd6308d3cb2e383d6b185abd16fa1958bf6bc54/Canon_5DSR_1080p.zip) | 22,871,037,140 | 22,870,444,351–22,871,037,041 | 592,691 |
| [Nikon_D810_1080p.zip](https://huggingface.co/datasets/presencesw/INTEL-TAU/resolve/9cd6308d3cb2e383d6b185abd16fa1958bf6bc54/Nikon_D810_1080p.zip) | 30,415,857,120 | 30,415,070,363–30,415,857,021 | 786,659 |
| [Sony_IMX135_BLCCSC_1080p.zip](https://huggingface.co/datasets/presencesw/INTEL-TAU/resolve/9cd6308d3cb2e383d6b185abd16fa1958bf6bc54/Sony_IMX135_BLCCSC_1080p.zip) | 22,580,796,457 | 22,580,128,496–22,580,796,358 | 667,863 |

Advertised LFS SHA-256 identifiers, not locally verified whole-archive hashes:

```text
Canon c96ea06f0859ffb96589d25cb3c8cfdecf364bfe63614ddcf1319e609c76c80b
Nikon b9e8f6e99aa933067898174fa104a429e00f55a09e674f399688563e1a3d3c10
Sony  42885f87e02f2d193951ee36a85329a1a634c7578f8b608119aaf6d6a4a44856
```

## Members, GT and CCM

The mirror directories contain 2,109 Canon, 2,793 Nikon and 2,120 Sony TIFFs, each with one matching `.wp`; these sum to the publisher's 7,022 scenes. Each camera also has four category directories. Camera-unique `field_1_cameras` contributes 1,645 Canon, 2,329 Nikon and 1,656 Sony scenes before the 30 Sony exclusions.

Exact member templates, with the actual numeric suffix retained in each frozen manifest:

```text
Canon_5DSR/field_1_cameras/C_5DSR_field_{id}.tiff
Canon_5DSR/field_1_cameras/C_5DSR_field_{id}.wp
Nikon_D810/field_1_cameras/N_D810_field_{id}.tiff
Nikon_D810/field_1_cameras/N_D810_field_{id}.wp
Sony_IMX135_BLCCSC/field_1_cameras/S_IMX135_BLCCSC_field_{id}.tiff
Sony_IMX135_BLCCSC/field_1_cameras/S_IMX135_BLCCSC_field_{id}.wp
```

The [publisher's dataset description](https://researchportal.tuni.fi/en/datasets/intel-tau/) identifies the processed TIFFs as black-corrected and saturation-normalized, and `.wp` as normalized camera-RGB white points. These are source claims until independently decoded. **No `.ccm`, `.mat`, JPEG or other non-directory files occur in these three 1080p archives.** The original description associates CCMs and frame metadata with raw Bayer packages. Their presence within raw archives was not verified because the original download path did not honor Range. A CCM is unnecessary for the strict camera-blind track and, if later obtained, must not enter its inference.

## Frozen selection and transfer sizes

Within each camera's eligible TIFF paths, sort ascending by hexadecimal SHA-256 of the UTF-8 string **`luma-cc-v2-fresh:` plus the exact archive-relative TIFF path**, then take the first 128 or 256. Never use GT, visual appearance, compressed size or model performance to rank candidates. Exclude all `field_3_cameras`, `lab_printouts`, `lab_realscene`, and Sony field 001–030. This hash sampling broadens coverage relative to a contiguous filename prefix, but does not establish statistical population representativeness.

All six metadata-only candidate manifests are retained under `docs/data/provenance/cc_v2/`, including each TIFF/GT path, compressed/uncompressed lengths, CRC32, local-header offset and exact combined image-plus-GT record range. The 256 candidates are alternatives; only the three 128 candidates were authorized for this acquisition. Hashes are in the camera `.summary.json` files and the downloader's frozen constants.

| Camera | Eligible field scenes | 128 TIFF+GT compressed bytes | Exact 128 range bytes with headers | 256 TIFF+GT compressed bytes |
|---|---:|---:|---:|---:|
| Canon 5DSR | 1,645 | 1,501,696,223 | 1,501,716,037 | 3,000,138,202 |
| Nikon D810 | 2,329 | 1,447,929,796 | 1,447,949,650 | 2,905,362,034 |
| Sony IMX135 BLCCSC | 1,626 | 1,483,995,708 | 1,484,019,872 | 2,964,466,008 |
| Total | 5,600 | **4,433,621,727** | **4,433,685,559** | **8,869,966,244** |

Thus an individual 128-scene camera subset fits the original 2 GB target. The authorized three-camera option fits 4.5 GB successful transfer; the two new Canon/Nikon cameras alone require 2,949,665,687 bytes with record headers.

Compressed TIFF size distributions (bytes; p25/p75/p95 use sorted index `floor(p*(N-1))`; p50 is the usual median):

| Camera | Min | p25 | p50 | p75 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Canon | 9,333,121 | 11,123,999 | 11,789,310 | 12,330,364 | 12,823,660 | 13,755,795 |
| Nikon | 8,128,868 | 10,723,629 | 11,258,276 | 11,953,441 | 12,735,653 | 14,270,978 |
| Sony, exclusions applied | 7,804,644 | 11,051,261 | 11,523,548.5 | 12,079,778 | 12,972,970 | 14,136,077 |

## Acquisition and verification

Reusable command:

```powershell
python scripts/download_cc_v2_fresh.py --data-root data/public/intel_tau_v2 --workers 4
python scripts/download_cc_v2_fresh.py --data-root data/public/intel_tau_v2 --verify-only
```

The downloader checks frozen manifest hashes, exact HTTP 206/Content-Range/body lengths, central/local ZIP header agreement, safe member paths, bounded decompression, CRC32 and SHA-256. It writes TIFF/GT bytes without decoding or interpreting them. Existing files are rehashed and CRC-checked before skipping. The entire planned successful range transfer is capped at 4.5 GB; a durable reservation journal caps cumulative attempted transfer, including retries and restarts, at 6 GB. Four workers may download concurrently. Images remain in ignored `data/public/intel_tau_v2/`; only provenance and hashes belong in version control.

The final [download report](provenance/cc_v2/download_report.json) records successful counts, transfer bytes, every file's SHA-256/CRC and failures. Scientific decoding and model evaluation are separate downstream steps after the model/protocol configuration is frozen. No image-content inspection, GT-value inspection or model-error calculation is part of this acquisition task.

**Measured completion:** 2026-09-10 18:26:38 UTC; 384 images and 384 GT files, **4,433,685,559 HTTP response-body bytes**, **4,658,541,806 uncompressed file bytes**, 805.44 seconds, zero failed attempts and zero final failures. All 384 responses were exact HTTP 206 ranges. The subsequent [local verification report](provenance/cc_v2/verification_report.json), completed at 18:26:54 UTC, re-read all 768 files and matched sizes, CRCs and the acquisition report's SHA-256 hashes. The utility also passed focused synthetic ZIP/path/header/CRC rejection checks and Ruff. These checks concern acquisition integrity only; no scientific evaluation result is implied.
