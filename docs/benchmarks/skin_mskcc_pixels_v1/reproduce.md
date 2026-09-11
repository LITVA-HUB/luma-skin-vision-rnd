# Reproduction of the local pixel experiments

Use the repository's pinned environment, original metadata/image acquisition
scripts and subject manifest. This is a local protocol, not the authors'2026
five-fold/pretrained protocol. Preserve the supplied reports as reference
evidence in a fresh checkout before running commands that write the same
benchmark directories; training deliberately refuses existing run/output
directories. Do not overwrite active research evidence.

Required source preparation, in order:

```powershell
.venv/Scripts/python.exe scripts/skin_mskcc_acquire_metadata.py
.venv/Scripts/python.exe scripts/skin_mskcc_data.py
.venv/Scripts/python.exe scripts/skin_mskcc_acquire_images.py
.venv/Scripts/python.exe scripts/skin_mskcc_pixels.py
```

On a fresh reproduction with the supplied v1/v2 benchmark directories retained
under distinct reference names, the output directories below must not exist:
`experiments/runs/skin_mskcc_pixel_controls_v1`,
`experiments/runs/skin_mskcc_pixels_v1`,
`experiments/runs/skin_mskcc_pixel_ablation_v2` and the per-model benchmark
subdirectories. The cache command creates the new v1benchmark root.

```powershell
.venv/Scripts/python.exe scripts/skin_mskcc_pixel_controls.py
.venv/Scripts/python.exe scripts/skin_mskcc_train_pixels.py
.venv/Scripts/python.exe scripts/skin_mskcc_pixel_report.py
.venv/Scripts/python.exe scripts/skin_mskcc_train_ablation_v2.py
.venv/Scripts/python.exe scripts/skin_mskcc_ablation_report_v2.py
.venv/Scripts/python.exe scripts/skin_mskcc_fusion_probe.py
.venv/Scripts/python.exe scripts/skin_mskcc_profile_pixels.py
```

The default neural runs use seeds17/29/43 and80epochs. Run GPU training and
profiling sequentially. Both checkpoint audits verify code/protocol/weight
hashes, best/final prediction replay and independently evaluated CIEDE2000.
The code enforces source-only cache roles. Calibration and final-test references
are not decoded. Raw images, pseudonymous participant mappings, checkpoints
and per-image arrays remain local/ignored; aggregate results and hashes are
committed. The rejected FP32cache was a preparation bug caught before fitting,
not an alternate model/data-selection trial; the verified FP64cache is used.
