# Smartphone coverage, 2026-09-11

Acquisition completed:47 selected scenes,560 records,3,229,998,120 payload
bytes. Every requested record passed CRC/size/SHA verification; the full160GB
archive was not downloaded. [Completion receipt](provenance/mobile_screen_2026_09_11/verification_all.json).
Three TRAIN scenes were used for loader development; all44 reserved paired-phone
test scenes remain numerically untouched. A separate model/weight lock is still
required before their first score. Acquisition does not establish phone accuracy.

The user explicitly prioritised iPhone and Android photographs. Canon-only
source training is a convenient real-GT development protocol, not evidence of
phone deployment. Earlier INTEL-TAU includes mobile Sony IMX135 sensor data,
but that is not a test of contemporary iPhone/Android JPEG/HEIC processing.

## Immediate Android candidate: Beyond RGB

Original [author repository](https://github.com/shirawerman/Beyond-RGB), pinned
0a3ed93e37f9f1b67fb23781c5e6ab34631020af, links its own
[Zenodo release](https://zenodo.org/records/16848482). The official API declares
CC BY4.0; creators match the paper's authors and Samsung research organisations.
This is an author-linked dataset release, not a third-party license badge.
Dataset use is cleared for R&D and commercial training under attribution terms;
repository code has no separately verified blanket license and is not imported.

The cameras are Samsung Galaxy S21 Plus SM-G996B and Oppo Find X5 Pro CPH2305,
alongside a16-channel research sensor. Use only the ordinary phone RGB captures
at inference. Camera-RAW captures are supplied as HDF5 with metadata, plus captures with
and without an X-Rite chart, chart coordinates and measured illuminant spectra.
[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Glatt_Beyond_RGB_A_Real_World_Dataset_for_Multispectral_Imaging_in_WACV_2024_paper.pdf),
[author structure](https://github.com/shirawerman/Beyond-RGB#beyond-rgb).
Loader audit now confirms the inspected HDF5 files are already demosaiced HWC
float32 camera RGB, with sampled values on a1/255 grid; do not treat them as
Bayer mosaics or apply the metadata10-bit scale again. CCM/black/white/CFA tags
exist. Reference saturation and dark-patch disagreement require a fixed quality
rule before evaluation. [Measured loader findings](phone_loader_findings.md). Illuminant
spectra plus uncalibrated camera RGB alone do not justify a skin DeltaE target.

The original split archive totals160,286,004,863 bytes. **Do not download it in
full.** Verified HTTP206 access permits sparse extraction. A4,654,108-byte
central-directory suffix lists30,888 entries. Local metadata audit finds1,686
scene folders with NT entries, including477 field-scene folders; these counts
describe the inspected archive, not additional independent scenes. The published
lists include372/46/47 field scenes in train/val/test; one training path is not
an exact archive match. Lab paths differ from release folder naming and will not
be guessed or used in this first phone screen.

Acquisition plan, before decoding any new image/GT values: use the first three
complete field scenes in a SHA256 ordering of the official training list for
loader development. Reserve every complete paired-phone field scene from the
official47-scene test list for later evaluation. Completeness means each camera
has NT/WT HDF5, NT/WT tags and WT chart coordinates, plus the scene spectrum;
include available privacy masks. No selection by illuminant, image appearance,
prediction or error. Save the exact selection and byte budget before download.
Hard payload cap4GB, total request traffic cap5GB. Verify local headers, sizes,
CRC32 and SHA256. Acquire no multispectral images. Do not decode held-out GT or
images for architecture selection. Both cameras from a scene remain in the same
role; this is paired phone evaluation, not a claim of scene-independent
leave-one-camera-out training. These test scenes were not used in V1–V5 training.

Selection is now frozen:3 loader scenes plus44 paired-camera reserved test scenes,
88 test input photographs and their separate chart-bearing references. Excluded
original test scenes10/264/330 lack required paired records. Metadata-derived
payload upper bound3,230,511,555 bytes. [Exact lock](provenance/mobile_screen_2026_09_11/beyond_rgb_selection.json),
SHA25686503dc8d2fdb247a57376c5a94ce5e0515003294b61286653b1e68099c0d38d.
This is an acquisition population, not a completed measured phone result.

## Other investigated phone sources

| Source | Phone / real reference | Rights and decision |
|---|---|---|
| [S24 Raw-sRGB](https://github.com/SamsungLabs/time-aware-awb/tree/main/s24-raw-srgb-dataset) |3,224 Galaxy S24 Ultra captures; chart-neutral illuminant, separate preference WB, RAW and in-camera Pro-mode JPEG | Root [CC BY-NC-SA4.0](https://github.com/SamsungLabs/time-aware-awb/blob/main/LICENSE.md). RESEARCH EVALUATION ONLY where separately justified; no product training or automatic assumption that commercial R&D evaluation is noncommercial. Not downloaded. |
| [LSMI](https://github.com/DY112/LSMI-dataset) |7,486 images/2,762 mixed-light scenes; Galaxy Note20 Ultra, Sony a9, Nikon D810; chart-derived local illuminants | Original dataset CC BY-NC4.0. Same commercial restriction; not downloaded. |
| [Rendered WB Set2](https://yorkucvil.github.io/projects/public_html/sRGB_WB_correction/dataset.html) |2,881 rendered images, including iPhone7, Galaxy S6 Edge, LG G4, Google Pixel and one DSLR; chart-based Adobe targets | Original site limits provision to reasonable academic fair use. Not commercially cleared, not acquired. Multiple renderings are not independent captures; Adobe targets are not ordinary phone JPEGs. |
| [Flash/Ambient](https://yaksoy.github.io/faid/) |iPhone6s/7 RAW+JPEG flash/ambient pairs | CC BY-NC-SA4.0, research only; flash pairing alone is not measured intrinsic surface-color GT. Not acquired. |
| [TCC](https://www.cs.ubc.ca/research/tcc/datasets/) |600 mobile sequences;400/200 split; PNG~14GB | Original data rights and exact phone still require verification. LICENSE UNCLEAR; no download. Single-frame protocol must remain sequence-disjoint. |
| [Nixon smartphone colorimetry replication](https://doi.org/10.7910/DVN/JJRH4N) |Real colorimetric study | Original Dataverse CC0, but inspected release contains four tabular result files, no camera-image benchmark. Does not solve the image-training need. |
| [Single Pixel Spectral CC](https://doi.org/10.1007/s11263-023-01867-x) |Huawei Mate20 Pro and measured spectra | Authors explicitly state collected data/code cannot be made public for business reasons. DO NOT USE as an available public dataset. |

## Acquired small iPhone repeatability set

The [original Dryad dataset](https://doi.org/10.5061/dryad.z8w9ghxg9) accompanying
[Zhang et al., PLOS ONE2023](https://doi.org/10.1371/journal.pone.0287099) explicitly
identifies iPhone SE2 and XS Max. Its linked Zenodo/Dryad release declares CC0.
The76,450,555-byte archive was downloaded and matched to the published MD5;
SHA25692ace21f94a652e585e95a51047223c7b7f5c0310929e6c2b510d333963f052f.
The3_objects directory contains15 SE2 and15 XS Max input PNGs, plus two already
corrected illustrative outputs. Those two outputs are not independent input
photos or a measured physical-color GT. The archive also contains synthetic
experiments, pH experiments and reference-board assets: they remain separate.
[Acquisition receipt](provenance/mobile_screen_2026_09_11/iphone_colorimetry_acquisition.json).

Category: CLEARED FOR R&D AND COMMERCIAL MODEL TRAINING under CC0, **but scientific
target suitability is limited**. Initially use the30 real object photos only as
a repeatability/ingestion stress set after checking correspondence, transformations
and reference-board leakage. No instrument-paired illuminant/absolute color target
was verified. Lower cross-image color variance alone is not sufficient: a model
that collapses all object colors can trivially lower variance. Do not optimise
primarily to that proxy or describe it as an illuminant/skin-DeltaE benchmark.
No images have yet been decoded or entered training. The authors' disclosed
provisional patent is not adopted by downloading the CC0 data; no code imported.

Further iPhone search remains active. Lack of a cleared, adequate iPhone accuracy
benchmark remains a gap, not a reason to infer iPhone performance from Android.
Evaluate RAW/linear and rendered JPEG/HEIC separately. Camera model, lens,
capture mode and ISP/rendering version can matter more than the operating-system
label. Skin-specific reference measurements remain future work.
