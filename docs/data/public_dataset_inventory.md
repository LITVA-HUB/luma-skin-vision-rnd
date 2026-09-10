# Public color-constancy dataset inventory

Reviewed 2026-09-10 against primary author pages and original dataset registration metadata. Status labels concern data rights, not scientific suitability or download completion. All sizes below are publisher estimates unless explicitly identified otherwise. No proprietary instrument-paired facial dataset is available or planned for this milestone.

## Decision

Use SimpleCube++ for model training, validation, risk fitting and calibration. INTEL-TAU supplies a separate camera-shift evaluation source. Its original registration declares CC BY-SA 4.0, now directly confirmed through the original Metax V3 API. The v1 milestone used C5Sony30 because original file discovery was then unavailable. The later v2 acquisition obtained 384 prespecified field scenes from a pinned mirror, as detailed below; neither milestone is a full INTEL-TAU evaluation.

The historical v1 pilot contains 30 Sony IMX135 examples redistributed by the **C5 paper authors**, whose README identifies them as INTEL-TAU. This permits a preliminary unseen-camera check under the upstream data license, not a representative camera-generalization claim. Keep these examples strictly test-only. Their processing provenance is less complete than that of the original archive. Their measured bit depth/shape and separate C5 code-license basis are retained below and in the license ledger.

## Main candidates

| Dataset | Coverage and representation | Ground truth and metadata | Download / size | Category |
|---|---|---|---|---|
| [SimpleCube++ / Cube++](https://github.com/Visillect/CubePlusPlus) | Simple: 2,234 downscaled 16-bit PNG images, calibration cube cropped; full: 4,890 RAW scenes. Canon 550D / 600D use the same sensor type. JPEGs are previews. | Simple RGB chromaticity CSV and semantic properties; full dataset adds EXIF, cube coordinates, multiple triangle GTs and CR2. No verified calibrated camera-to-XYZ matrix in Simple. | [Simple ZIP](https://zenodo.org/record/4153431/files/SimpleCube++.zip), about 2 GB. Full approximately 200 GB: outside budget. | CLEARED FOR R&D AND COMMERCIAL MODEL TRAINING |
| [INTEL-TAU](https://researchportal.tuni.fi/en/datasets/intel-tau/) | 7,022 scenes, Canon 5DSR, Nikon D810, Sony IMX135. Publisher describes 1080p linear TIFFs with black subtraction and saturation normalization; raw Bayer also available. | Chart-measured white points, normalized RGB `.wp` for processed images; raw uses R/G,B/G. Acquired 1080p archive directories contain only TIFF/GT pairs, no CCMs. Raw package describes `.ccm`, frame metadata and spectral responses; raw member availability remains unverified. Faces/plates masked by authors; chart images withheld. | [Original Fairdata record](https://etsin.fairdata.fi/dataset/f0570a3f-3d77-4f44-9ef1-99ab4878f17c). Original multipart totals: Canon 22,871,037,140 B; Nikon 30,415,857,120 B; Sony BLCCSC 22,580,796,457 B. V2 sparse mirror subset: 384 TIFF/GT pairs, 4,433,685,559 transfer bytes. | CLEARED FOR R&D AND COMMERCIAL MODEL TRAINING under original terms; this project's INTEL-TAU use remains evaluation-only |
| [NUS-8](https://yorkucvil.github.io/projects/public_html/illuminant/illuminant.html) | 1,736 images / 8 cameras: Canon 1Ds MkIII, 600D; Fujifilm X-M1; Nikon D5200; Olympus E-PL6; Panasonic GX1; Samsung NX2000; Sony A57. Extra Nikon D40 set is separate. RAW, linear PNG and preview JPEG. | Achromatic chart-patch illuminants; patch masks, ROI, black and saturation levels in MAT. Original README does not supply calibrated XYZ CCMs. | Author page links per-camera ZIP parts and MAT files through Sync. Archive byte sizes not verified. | LICENSE UNCLEAR |
| [Gehler-Shi](https://www.cs.sfu.ca/~colour/data/shi_gehler/) / RECommended | 568 images, Canon 1D / 5D. Shi rendering is linear camera RGB in PNG, preserving 12-bit counts; original processed TIFFs are nonlinear. | Macbeth chart positions / illumination labels; black subtraction remains necessary for Shi images. RECommended replaces GT, not photographs. Calibrated XYZ CCM not verified. | SFU provides four PNG archives, each 1–2 GB. [UEA thumbnails](https://colour.cmp.uea.ac.uk/datasets/reprocessed-gerler.html): 549×365 TIFF, approximately 1 MB/image. | LICENSE UNCLEAR |

## INTEL-TAU download details and limitations

Original registration: `urn:nbn:fi:att:f8b62270-d471-4036-b427-f21bce32b965`. **The Etsin dataset UUID differs from the URN suffix:** `f0570a3f-3d77-4f44-9ef1-99ab4878f17c`.

The publisher's prose lists `Canon_5DSR_1080p.zip`, `Nikon_D180_1080p.zip`, `Sony_IMX135_BLCCSC_1080p.zip`, isotropic variants and `splits.zip`. Historical v1 discovery did not obtain original URLs or byte counts: Etsin/Metax timed out locally, and web retrieval returned a JavaScript shell. The later original file records resolve the prose's `D180` typo to actual `Nikon_D810` filenames. A third-party mirror's MIT badge is **not** the data license and is not adopted here.

For a pilot only, [C5 author README](https://raw.githubusercontent.com/mahmoudnafifi/C5/main/README.md) identifies its `images/` examples as Sony INTEL-TAU and its JSON field `illuminant_color_raw` as the GT. Git ref observed via `git ls-remote`: `dca92361d43c8e737697fb27ac6eb9aa5d640bc4`. GitHub contents API returned 30 PNGs and matching JSONs. Subsequent acquisition by the main milestone task measured **13,826,580 bytes total** and inspected images as **uint16, 256×384**. These are the derived C5 samples, not original-resolution INTEL-TAU files.

File template at the observed immutable ref, index `001` through `030`:

```text
https://raw.githubusercontent.com/mahmoudnafifi/C5/dca92361d43c8e737697fb27ac6eb9aa5d640bc4/images/S_IMX135_BLCCSC_field_{index}_sensorname_sony_imx135.png
https://raw.githubusercontent.com/mahmoudnafifi/C5/dca92361d43c8e737697fb27ac6eb9aa5d640bc4/images/S_IMX135_BLCCSC_field_{index}_sensorname_sony_imx135_metadata.json
```

The first JSON was read and contains a three-component target. Images were downloaded by the main milestone task, separately from this inventory review. Sampling, transformations and possible scene dependence mean 30 examples are insufficient for a formal unseen-camera benchmark; no tuning, calibration or threshold selection may use their labels.

### Historical v1 original-host follow-up audit

A bounded follow-up used [official Metax V3 API documentation](https://metax.fairdata.fi/v3/swagger/) and [legacy V2 documentation](https://metax.fairdata.fi/v2/docs/v2/datasets.html) to verify these read-only discovery routes:

```text
https://metax.fairdata.fi/v3/datasets/f0570a3f-3d77-4f44-9ef1-99ab4878f17c
https://metax.fairdata.fi/v3/datasets/f0570a3f-3d77-4f44-9ef1-99ab4878f17c/files
https://metax-legacy.fairdata.fi/rest/v2/datasets/f0570a3f-3d77-4f44-9ef1-99ab4878f17c
```

The hosts still timed out on connection; web retrieval could not read the JSON endpoints. No usable browser was available for the JavaScript page. Consequently no original archive URL, checksum, ranged-ZIP capability or smaller original camera subset was verified. Do not replace this missing evidence with invented URL patterns.

At that v1 review, a third-party Hugging Face mirror's listing reported Canon 1080p ZIP 22,871,037,140 bytes, Nikon 1080p ZIP 30,415,857,120 bytes, and Sony BLCCSC 1080p ZIP 22,580,796,457 bytes. All exceed 20 GB individually. Those were then mirror-reported values without original-file confirmation. Its smaller transformed fold archive has undocumented preprocessing and was not adopted. The frozen v1 benchmark remains C5Sony30-only for the second camera.

### Measured v2 acquisition and current access status

The [original dataset endpoint](https://metax.fairdata.fi/v3/datasets/f0570a3f-3d77-4f44-9ef1-99ab4878f17c) and its paginated `/files` endpoint succeeded through PowerShell using normal system TLS trust. The original record directly confirms **CC BY-SA 4.0**. All 32 official file records and multipart SHA-256 checksums are saved in [v2 provenance](provenance/cc_v2/official_v3_files_verified.json), with the [second page](provenance/cc_v2/official_v3_files_page2_verified.json). Their multipart byte totals exactly match the three mirror lengths. Original Etsin download authorization succeeds, but the tested last Nikon part ignored an exact byte-range request and returned HTTP 200; its body was closed unread. Original archive-byte identity therefore remains unverified.

The pinned `presencesw/INTEL-TAU` mirror revision `9cd6308d3cb2e383d6b185abd16fa1958bf6bc54` supports exact HTTP 206. ZIP metadata was read first, and **128 camera-unique `field_1_cameras` TIFFs per camera** were selected by the lowest SHA-256 of `luma-cc-v2-fresh:` plus the archive-relative TIFF path. All lab/common-camera sets and Sony field IDs 001–030 were excluded. The frozen manifests contain exact members, paired GT paths, CRCs and offsets; no image or GT value was used for selection.

Measured acquisition: **384 TIFFs + 384 `.wp` files**, **4,433,685,559 response-body bytes**, **4,658,541,806 uncompressed bytes**, **805.44 seconds**, no retries or failures. Every member passed local/central ZIP-header, length and CRC checks, followed by a full local reread matching all 768 SHA-256 records. [Acquisition procedure and source limitations](cc_v2_camera_acquisition.md), [download report](provenance/cc_v2/download_report.json), [verification report](provenance/cc_v2/verification_report.json). No images or GT values were decoded by the acquisition task; source/decoding review and scientific evaluation are separate steps.

All newly acquired INTEL-TAU data are **evaluation-only by project protocol**; this is not a noncommercial license restriction. Their labels, target-set statistics and camera-specific information must not fit weights, thresholds, normalization or selection rules. Weights remain trained only from **CC BY 4.0 SimpleCube++**. The HF MIT badge is ignored for image rights. Member CRCs and local SHA-256s establish pinned-mirror consistency, not equivalence to the original multipart checksums. This 384-scene subset improves camera coverage but does not establish full-dataset, person-independent or instrument-paired facial validation.

## Alternatives screened

| Dataset | Real data, cameras, GT / metadata | Size and download | Category / suitability |
|---|---|---|---|
| [MLS — Multiple Light Source](https://github.com/visillect/mls-dataset) | 24 object scenes ×18 illumination configurations; Canon 5D MkIII; linear hardware-RGB PNG, raw mosaics. Surface masks, measured reflectance/illuminant spectra; approximate custom-sRGB transform. Camera sensitivity CSV is digitized from a separately cited paper. | `ftp://vis.iitp.ru/mls-dataset/images_16bit_color.zip` 2 GB; `masks_16bit_color.zip` <1 MB; `illuminants.zip` 4 MB; `surfaces.zip` <1 MB. | CLEARED FOR R&D AND COMMERCIAL MODEL TRAINING for author-created data. No ready single RGB-GT CSV. At most a planned 24-scene `2HAL` pilot after target validation; mixed sources need spatial evaluation. |
| [MIT Multi-Illumination](https://projects.csail.mit.edu/illumination/) | >1,000 indoor scenes ×25 lighting directions; Sony RAW plus HDR EXR/JPEG. Real lighting variation; no verified benchmark camera-RGB chromaticity CSV. | [EXR test](https://data.csail.mit.edu/multilum/multi_illumination_test_mip2_exr.zip) 1.5 GB; [JPEG test](https://data.csail.mit.edu/multilum/multi_illumination_test_mip2_jpg.zip) 190 MB. JPEG train 6.6 GB; EXR train 50 GB; RAW >2 TB, with per-scene links. | CLEARED FOR R&D AND COMMERCIAL MODEL TRAINING. Paired lighting robustness candidate; not a drop-in angular-illuminant benchmark. |
| [LSMI](https://github.com/DY112/LSMI-dataset) | 7,486 images / 2,762 scenes; Galaxy Note20 Ultra, Sony α9, Nikon D810. RAW/sRGB; chart-derived illuminants, mixture maps, scene splits and metadata. | Author download form; total bytes not verified. | RESEARCH EVALUATION ONLY. CC BY-NC 4.0. No commercial training or commercial-purpose evaluation assumed. |
| [Shadows & Lumination](https://ilijad.github.io/shal.html) | 2,500 two-illuminant scenes / five cameras; Canon 5D, 550D, Sony α300, Panasonic FZ1000, Motorola One Fusion+. Two GT illuminants and binary regions. | Author points to `http://bit.ly/shal_dataset`; bytes/terms unresolved. | LICENSE UNCLEAR. No explicit original data grant found on inspected author page. |
| [MIMI](https://github.com/Spectricity/MIMI-dataset) | 773 scenes, RGB+multispectral camera pair, chart/no-chart captures, white-point and color references. | Download request form, publisher states 319.4 GB single ZIP. | RESEARCH EVALUATION ONLY. CC BY-NC 4.0; also outside budget. |
| [Mixed-illuminant WB test set](https://github.com/mahmoudnafifi/mixedillWB) | SYNTHETIC: 150 rendered mixed-illumination images and corrected targets. | Author links 8-bit JPG / 16-bit PNG; bytes not verified. | RESEARCH EVALUATION ONLY. Explicit research-only data restriction; not real-camera validation. |
| [Sony DoLP color constancy](https://github.com/sony/dolp-colorconstancy) | Polarization-specific RAW-RGB and chart targets; counts not audited. | Requires Microsoft account and contacting authors; no direct open archive obtained. | LICENSE UNCLEAR. MIT statement explicitly covers software, not data. Excluded from acquisition scope. |
| Original INTEL-TUT release | Predecessor withdrawn for privacy problems, according to [INTEL-TAU paper](https://arxiv.org/abs/1910.10404). | Do not retrieve old mirrors. | DO NOT USE. Use privacy-masked INTEL-TAU instead. |

## Scientific boundary

## Explicitly excluded product dependencies

| Dataset | Original terms checked | Data and feasibility | Decision |
|---|---|---|---|
| Rendered WB | [Author WB_sRGB repository, Commercial Use](https://github.com/mahmoudnafifi/WB_sRGB#commercial-use) restricts software AND dataset to research; commercial applications require a separate license. | 105,638 rendered sRGB images in expanded release; paired rendered corrected targets and metadata. Not independent physical facial measurements; archive sizes not acquired. | RESEARCH EVALUATION ONLY; not downloaded, no production training/teacher dependency. |
| FaceOLAT | [Author processing repository](https://github.com/prraoo/FaceOLAT) states academic/research purpose and refers to the [access site](https://gvv-assets.mpi-inf.mpg.de/FaceOLAT/) for terms. Full agreement remains behind registration; no account created. | 139 subjects, 40 cameras, 350 OLAT conditions, 4K capture, roughly9TB; color-calibrated AVIF from RED RAW. Not a ready instrument-paired cheek Lab dataset. | LICENSE UNCLEAR for this commercial R&D use; academic/research intention is not commercial clearance. Not downloaded. |

## Scientific boundary (continued)

These datasets support illumination estimation and limited camera-shift experiments. They do not supply instrument-paired facial-region CIELAB D65/2° measurements, certify skin-color accuracy, establish person-level coverage, or justify uncertainty/abstention coverage on real faces. A camera-RGB illuminant estimate is not an absolute Lab target. A CCM alone does not establish an instrument-traceable calibration.

Keep original scene IDs and related captures together. Separate Cube/Cube+/Cube++ relatives before splitting; SimpleCube++ is not an independent dataset from Cube++. Holding out 550D versus 600D is weak evidence of unseen-sensor generalization because the authors identify the same sensor type. Record preprocessing, masks, camera ID and target normalization independently for every adopted source.
