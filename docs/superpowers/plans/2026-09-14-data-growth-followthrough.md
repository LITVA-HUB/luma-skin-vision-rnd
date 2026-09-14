# Luma data growth follow-through

> Execute inline using the existing executing-plans workflow. User authorized autonomous data acquisition and experiments; no new approval is needed.

**Goal:** finish the Seg1 component evidence and expand the measured facial reflectance TRAIN corpus from 3 to 19 source faces.

**Architecture:** preserve all frozen studies and original 19/5/5 UMINHO assignments. Store additional originals and an acquisition plan on D. Reuse the three original cubes by verified path, without recounting them as new data. Verify source size/MD5 plus local SHA256 before decoding. No validation/test cubes are acquired in this extension.

**Tech stack:** existing Python environment, urllib/hashlib, NumPy/SciPy, existing React dashboard. No new dependencies.

**Spec:** latest user requests suitable datasets and quality experiments regardless of unclear licenses; the supplied photos have no measured color reference. Native instrument color and image-derived apparent color remain separate targets.

- [x] Verify Seg1 source hashes, selected epoch, saved test metrics, apparent-color paired analysis and actual CPU timing. Report the limitations of each endpoint.
- [x] Test that UMINHO selection excludes held roles and unsafe filenames; implement `scripts/skin_uminho_train_expand_v2.py`, freeze all 19 source rows before new acquisition and validate every file.
- [x] Audit cube shapes, wavelength counts, numerical ranges and source-mask support one cube at a time. These are measured spectra, not independent RGB camera images or verified skin-only masks.
- [x] Check the completed dashboard in a real browser and update the current status and data report.
- [x] Record useful next color experiment and preserve the broad active goal. No native-phone quality claim follows from this component milestone.

Completed evidence: `docs/benchmarks/facial_skin_v1/report.md`; local Seg1 verification SHA3231d667a9091e05678fa59e3b1ba0f825ce2ea5f1d503df12465509d6dd01e3. UMINHO16new TRAIN cubes/1,738,512,850bytes;19total/1,975,360,986bytes,8,357,334foreground spectra. Five acquisition selection/path tests pass. All launched jobs in this follow-through exited0. Native-phone quality is still unvalidated; the broad goal remains active.
