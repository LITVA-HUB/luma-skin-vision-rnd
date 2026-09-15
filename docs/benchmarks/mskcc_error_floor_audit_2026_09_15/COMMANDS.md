# Executed commands — diagnostic audit, no model training

Working directory: luma-skin-vision-rnd; base9cad271aad159d73083259967b4107b2ce828eb1.
Relative output ../private_artifacts/error_floor. Use the same paths or substitute
one output directory consistently. Python3.12.14; requirements-audit.txt.

```bash
python scripts/error_floor/restore_metadata.py
PYTHONPATH=src python scripts/error_floor/instrument_audit.py --output ../private_artifacts/error_floor
python scripts/error_floor/restore_train_images.py
PYTHONPATH=src python scripts/error_floor/capture_audit.py --rows ../private_artifacts/error_floor/audit_train_rows.private.json --output ../private_artifacts/error_floor/capture --workers 2
PYTHONPATH=src python scripts/error_floor/reference_capture_counterfactuals.py --audit ../private_artifacts/error_floor
python scripts/error_floor/plot_audit.py --instrument ../private_artifacts/error_floor/instrument.json --capture ../private_artifacts/error_floor/capture/capture.json --output ../private_artifacts/error_floor/figures
python scripts/error_floor/build_report.py --audit ../private_artifacts/error_floor --output ../private_artifacts/error_floor/report
PYTHONPATH=src python -m unittest discover -s tests -p test_error_floor_c_audit.py
```

All listed analysis commands completed exit0. Unit tests:4synthetic checks, not
real C analysis. Separate real-feature alignment verified all966rows and no raw
source IDs in the anonymized NPZ. Original model/source/provenance files remain
unchanged according to git diff.

Recovery history: initial direct transport failed; the permitted default proxy
transport restored originals. An interrupted duplicate download produced one
staging rename error; after stopping it, final cache verification accepted
all966images,zero failures. Prior partial capture278images was only a smoke;
its metrics are excluded from this report. Final capture processed966images.

# NOT EXECUTED — requires missing saved C OOF

```bash
PYTHONPATH=src python scripts/error_floor/c_oof_audit.py inspect /path/oof_anonymized.npz
PYTHONPATH=src python scripts/error_floor/c_oof_audit.py run --oof /path/oof_anonymized.npz --prediction-key ACTUAL_C_KEY --target-key ACTUAL_TARGET_KEY --index-key ACTUAL_ROW_INDEX_KEY --group-key ACTUAL_GROUP_KEY --rows ../private_artifacts/error_floor/audit_train_rows.private.json --capture /path/data/capture_features_anonymized.npz --capture-anonymized --fixed-masks /path/data/diagnostic_subset_masks.anonymized.npz --output ../private_artifacts/error_floor/c_oof
```

The actual key names must be read from the original file. The analyzer validates
original indices and target/group/capture correspondence and refuses mismatches.
Do not fabricate predictions or label synthetic tests as C results.
