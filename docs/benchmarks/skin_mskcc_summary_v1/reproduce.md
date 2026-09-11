# Reproducing the direct-skin source pilot

Use a fresh checkout/output directory, the pinned Python environment in this
repository and original freely accessible MSKCC files. Windows commands from
repository root:

```powershell
.venv/Scripts/python.exe scripts/skin_mskcc_acquire_metadata.py
.venv/Scripts/python.exe scripts/skin_mskcc_data.py
.venv/Scripts/python.exe scripts/skin_mskcc_reproduce.py --output experiments/runs/mskcc_reproduction_1
```

The wrapper changes output directories only, runs the unchanged frozen fit
algorithm and performs its independent audit. It refuses an existing output
directory. Choose a new destination for another reproduction; committed
evidence stays intact. Numerical source labels are decoded only for TRAIN
and VALIDATION. Ten test and six calibration people remain reserved.

Image acquisition, independently from the summary-feature pilot:

```powershell
.venv/Scripts/python.exe scripts/skin_mskcc_acquire_images.py
```

This acquires only the1838instrument-paired original JPEGs, with original
per-image API license checks and S3 Content-Length / MD5 validation; it can
resume verified cached files. It does not decode test pixels or references.
The source-only summary experiment uses no image pixels or external pretrained
weights. It depends on authors' supplied image-median Lab features, and should
not be labelled a reproduced end-to-end image algorithm.

The result JSON contains all models, source roles, data/code/model hashes,
warning logs, modality/camera strata and fixed-coverage curves. Checkpoint and
prediction arrays remain in local ignored experiment storage, because they
carry dataset record/group identifiers. Regenerating them uses the same
original CC-BY source and deterministic grouping protocol.
