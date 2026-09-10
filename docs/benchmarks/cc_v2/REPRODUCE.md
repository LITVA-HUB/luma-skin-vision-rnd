# Reproduce CC v2 and audit the saved result

Run from the repository root in PowerShell. This milestone is local; datasets, caches, fitted `.pt`/`.joblib` and ONNX files remain outside Git. Committed per-image predictions/GT/errors, source selections, configuration histories, receipts and exact code snapshots permit inspection without downloading photographs. Full prediction/training reproduction additionally requires the legitimately acquired data and local artifacts described below.

## Environment and evidence identity

Python3.12.6, torch2.8.0+cu128, torchvision0.23.0+cu128, scikit-learn1.9.0, Windows11, RTX4060 8GB. Exact dependencies are in `uv.lock`; new research dependency licenses are archived. Do not update dependencies or format frozen source while auditing original artifacts.

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
uv sync --all-extras --frozen
.venv/Scripts/pytest -q
.venv/Scripts/ruff check .
.venv/Scripts/ruff format --check .
uv lock --check
```

Original training source hash: `598a44f190d624db58fd4d2a60368f59bf41b1f3e8620c8c49c55d6e1e680398`. The runner fingerprints all `src/**/*.py`, `pyproject.toml` and `uv.lock`. Selector/statistics scripts have their own bindings. [Exact-byte archive manifest](reproducibility/archive_manifest.json) maps137 metadata/code/license files (6,883,090 bytes) to originals, including all17 CNN configurations/training histories, selector choices, two statistics screens and the complete source snapshot. `.txt` suffixes on snapshots are archival only. Git attributes disable line-ending conversion for the new evidence trees; archived byte hashes must survive checkout unchanged.

Absolute Windows paths, mtimes, timestamped configurations and compressed-file container bytes are recorded provenance, not portable semantic identities. A clean retraining produces new artifacts/hashes and must have its own honest locks. Do not rewrite an old lock to make different weights appear to be the original run. No external preregistration is claimed; local lock ordering is evidence of this recorded workflow, not proof that no unrecorded experiment could exist.

## Inspect/recalculate measurements without neural inference

```powershell
.venv/Scripts/python scripts/cc_v2_report.py
.venv/Scripts/python scripts/document_cc_v2.py
.venv/Scripts/python scripts/cc_v2_figures.py
```

These consume the25 saved JSON files under `docs/benchmarks/cc_v2/runs/`. They rebuild the aggregate, full result CSV, standard metrics/calibration diagnostics, report tables and figures. They do not fit or run a model. Running them regenerates descriptive outputs; retain the original milestone when comparing outputs.

Paired bootstrap uses saved scores/errors plus grouping from the fresh manifest for camera/reference-hash strata. Preserve the original receipt; write a new output:

```powershell
.venv/Scripts/python scripts/cc_v2_bootstrap.py --first 'ccv2_sog_large_g0::combined' --second 'ccv2_direct_large_g0::combined' 'gw_ridge1::cheap' 'ccv2_sog_large_g0::context' --output artifacts/cc_v2_bootstrap_recheck.json
```

The same2,000 draws/seed1209/grouping preserve original mean/risk80 intervals; AURC intervals are a secondary descriptive extension. [Independent audit instructions](audit_readme.md) explain `scripts/audit_cc_v2_metrics.py`: this fuller audit needs cached GT/image arrays, prediction NPZs, local weights for hash verification, and Git7637d6d history. It recomputes metrics and verifies bindings; it does not run inference or unpickle model files. It is **not JSON-only**. Use a new receipt path; the script refuses overwriting its output. Saved original and committed-code recheck receipts agree exactly on all14 comparison fields.

## Acquire and prepare source data in a clean checkout

Follow the [original license inventory](../../data/public_dataset_inventory.md) and [V1 protocol](../../research/public_protocol_v1.md). The source is the original2.113GB SimpleCube++ archive, CC BY4.0; retain author attribution, ZIP checksum and preprocessing notes.

```powershell
.venv/Scripts/python scripts/download_public_cc.py
.venv/Scripts/python scripts/download_cc_v2_pilot_exact.py
.venv/Scripts/python scripts/prepare_public_cc.py
.venv/Scripts/python scripts/classical_public_cc.py
```

**Do not run preparation over an existing frozen cache.** The historical source preparation can overwrite `data/processed/cc128`; use a separate clean checkout/output population. The final source partitions are1126 train /119 validation /259 risk /268 calibration /462 previously observed official test. Model/risk/calibration groups are capture-date-disjoint; the official test is not date-disjoint from development. Neural inputs are128×128 FP16 cached linear thumbnails; classical experts use full processed images. Decoder metadata and target exclusion are documented in the original protocol.

The Sony30 regression pilot comes from C5's author redistribution under original INTEL-TAU BY-SA terms. It is required by the combined historical preparation script, although not by CNN training itself. Its historical downloader resolves a moving branch: the new exact-pilot helper above instead uses the60 pinned image/GT URLs and hashes in the committed pilot provenance. Existing files are verified; mismatches fail without overwrite. The pilot is separate from the new384-image test and never a training source.

## Fresh evaluation-only acquisition

[Frozen384-image acquisition](../../data/cc_v2_camera_acquisition.md) specifies original Metax terms, mirror revision, deterministic IDs, ZIP members/CRC/SHA, bandwidth limits and original-archive identity limitation. Preserve its three manifest hashes exactly. No Hugging Face license badge overrides the original CC BY-SA4.0 grant.

```powershell
.venv/Scripts/python scripts/download_cc_v2_fresh.py
.venv/Scripts/python scripts/download_cc_v2_fresh.py --verify-only
.venv/Scripts/python scripts/prepare_cc_v2_fresh.py
```

These commands belong in a clean replication workspace; acquisition writes receipts into the provenance directory, and preparation refuses an existing destination. The preparation requires an estimator-selection lock before pixel decoding. The historical lock is valid only as a historical record; any new training must create a new lock before its evaluation. This384-image population is now observed for the project and cannot become an independent new discovery set again.

The data contain128 camera-unique field images per Canon5DSR/NikonD810/SonyIMX135. The camera models are absent from fitting; Sony's model was known through the old pilot. Processed1920×1080 uint16 TIFF→RGB/65535, remove max≥0.98 pixels, no repeated black correction/gamma/CCM/chart masking, area128square, per-image exposure95th percentile normalization, clamp0–4, FP16 cache. Actual image hashes and WP labels remain separate. Sparse mirror extraction verifies pinned-mirror consistency, not identity to original multipart archive checksums.

## Source-only CNN experiment matrix

All runs:120epochs, batch32, AdamW lr0.001/weight decay0.0001, cosine2e-5, clip5, horizontal flips, scalar exposure augmentation, scratch initialization; see archived config/training histories for exact arguments. No target-domain aggregate statistic, image or GT enters fitting. Gain0.7 means matching random diagonal gain on real image and illuminant, not new physically measured examples.

| Backbone | Mode | Gain augmentation | Seeds |
| --- | --- | --- | --- |
| small | direct |0|17,29,43|
| small | direct |0.7|17|
| small | gw |0 and0.7|17 each|
| small | sog |0|17|
| small | sog |0.7|17,29,43|
| large | direct |0|17,29,43|
| large | gw |0|17|
| large | sog |0|17,29,43|

Example for a **new** run directory, never an existing original:

```powershell
.venv/Scripts/python -m luma_skin_vision.cc.v2_experiment train --out experiments/runs/replication_sog_large_s17 --mode sog --backbone large --gain-aug 0 --seed 17 --epochs 120 --batch 32
.venv/Scripts/python -m luma_skin_vision.cc.v2_experiment predict --out experiments/runs/replication_sog_large_s17
.venv/Scripts/python scripts/cc_v2_select.py fit --runs experiments/runs/replication_sog_large_s17
```

Checkpoint choice uses unaugmented source-validation mean reproduction error. Error-head choice uses5-fold capture-date GroupKFold on259 held-out risk examples: StandardScaler+Ridgeα1/10/100 or HGB3/7leaves. Choose pooledOOF risk80 then AURC; refit risk head, mean-scale calibrate on separate268 examples. Context/cheap/combined blocks are retained. Final historical source-only primary is SoG-large combined; matched direct-large combined is the strong C+; context-only is the standard C+.

## Standard statistics and historical controls

The current corrected statistics script fits12 source-only models (direct/GW × ridge1/10/100, HGB7/15, ExtraTrees128). Each mode has51features. Train on1126, select on119. Before any new-camera errors, a numeric fix made analytically zero GW features exactly zero; the original12-candidate screen remains archived, and both winner IDs were unchanged. Run corrected code to a **new** directory/summary:

```powershell
.venv/Scripts/python scripts/cc_v2_statistics.py train --out experiments/runs/replication_statistics --summary artifacts/replication_statistics_screen.json
```

Historical final controls are `direct_hgb7` and `gw_ridge1`, one fitted estimator each. `scripts/cc_v2_statistics_eval.py fit` requires an estimator lock matching the actual source screen; its separate evaluation requires a matching final head lock. Consult `statistics_fixed_selection_lock.json`, `final_head_lock.json` and the archived selections for the explicit schema. In a new study generate these records from the new artifacts **before** test errors. Do not pass old hash-bound locks to newly trained weights or disable checks.

Six V1 checkpoints are preserved (baseline/proposed×17/29/43). `scripts/cc_v2_legacy.py` authenticates them against commit7637d6d and old code/data; new context/combined heads fit only source-risk/calibration rows. Old selectors and their outputs remain separately labeled. This is a selector upgrade, not retraining V1 estimators. Original synthetic/V1 evidence is not overwritten.

## Frozen final evaluation and deployment checks

The historical final lock precedes new-camera errors and binds17 CNN selectors,6 legacy selectors,2 statistics selectors, scripts and fresh data. `scripts/run_cc_v2_evaluation.py` validates it before invoking CNN/legacy evaluation and refuses existing result files. It is an original-run driver, **not a generic new-study lock generator**. Statistics evaluation checks the final head lock itself. Do not invoke lower-level evaluators as a substitute for the outer lock preflight.

For an exact audit on the original local bundle, preserve original outputs and use a separate recheck directory with the same guarded evaluations. For clean retraining, issue a new source-only lock manifest for the new paths/hashes and retain that study's outputs separately; compare numerical metrics and uncertainty, not timestamp-dependent checkpoint/file identity. The existing lock cannot be honestly reused as a pre-evaluation lock for new choices.

After the positive matched comparison, the representative seed17 was profiled and exported:

```powershell
.venv/Scripts/python scripts/profile_cc_v2.py profile
.venv/Scripts/python scripts/profile_cc_v2.py export
```

The profile supports a new `--output` destination. Export binds the original representative run and refuses its existing `export_v2` directory; preserve that directory and use a separate copied checkout/bundle for an exact export recheck. Do not delete original artifacts to make the command rerun. See [original profile/export receipts](profiles/README.md). The graph takes preprocessed3×128×128 input and outputs illuminant, expected reproduction error, validity and source-threshold acceptance. Byte decoding remains external. ONNX CPU numerical parity/invalid refusal passed; TensorRT, quantization and ONNX latency were not measured. PyTorch model/full-PNG timings and allocator-only memory have explicit scope in the profile.

No sequence above validates facial skin color. Do not tune on the observed384 test images and then present them as a new independent confirmation.
