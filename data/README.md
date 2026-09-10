# Data directory

Participant images and instrument exports do not belong in Git. Keep them under an explicitly supplied local data root and use relative `image_path` values that cannot escape it. This repository may contain only non-sensitive schemas and clearly labeled synthetic demo fixtures.

Real manifests contain one JSONL row per image × cheek and must follow `src/luma_skin_vision/data.py::Record`. Do not mix `INSTRUMENT`, `PHOTO_REFERENCE`, and `SYNTHETIC` rows in a manifest. Preserve image bytes, SHA-256 hashes, calibration/repeat/timestamp provenance, and subject-disjoint splits.

Read `docs/data/dataset_protocol.md`, complete `docs/data/DATA_REQUIRED.md`, and fill `docs/data/data_card_template.md` before collection. Never upload participant images to external services or encode direct identifiers in `subject_id`.

