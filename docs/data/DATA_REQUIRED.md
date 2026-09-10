# DATA_REQUIRED handoff

No real participant or instrument data is present. To begin the pilot, provide a local dataset root containing original image files, one UTF-8 JSONL manifest, the auxiliary capture/registration logs, consent status records separated from identity linkage, and the completed data card.

## Decisions to lock before capture

- Instrument manufacturer/model and serial number.
- Actual aperture in mm, measurement geometry, and `SCI` or `SCE` mode.
- D65/2° output/conversion procedure and calibration procedure.
- Three camera/device/module pipelines and four named lighting setups.
- Fixed acclimation period, capture distance/pose tolerance, image settings, and invalid-repeat rule.
- Repeatability acceptance tolerance, set from instrument capability and operational need before model results are viewed.
- Opaque-ID generation, consent text, local access controls, retention/deletion policy, and withdrawal process.
- Subject allocation seed and the unseen device/light conditions, locked before training.

## Pilot quantity and structure

Collect 10–15 consenting people × 3 camera pipelines × 4 lighting conditions × 2 independent image repeats. Measure both defined cheek sites with at least three instrument readings per site per session. This pilot diagnoses feasibility, repeatability, registration quality, and variance; it is not assumed to establish product performance.

## JSONL record contract

Create one row per image × cheek region. Two rows for the same image share image/capture fields. Every field below maps directly to `src/luma_skin_vision/data.py::Record`:

| Field | Required handoff value |
|---|---|
| `schema_version` | Literal `1.0`. |
| `data_kind` | `INSTRUMENT` for physical instrument targets. Do not mix kinds in one manifest. |
| `subject_id` | Opaque `[A-Za-z0-9_-]+` identifier; no direct or inferred identity content. |
| `session_id` | Stable visit/session identifier. |
| `image_id` | Unique logical image identifier, consistent across the two cheek rows. |
| `image_path` | Relative path under the manifest directory/data root; never absolute or escaping the root. |
| `image_sha256` | Lowercase SHA-256 of the unchanged image bytes. |
| `image_mirrored` | Required boolean: true if post-EXIF stored pixels need horizontal unmirroring. Verify with an asymmetric registration frame. |
| `device_manufacturer`, `device_model`, `camera_module`, `front_or_rear` | Actual capture pipeline; `front_or_rear` is `front` or `rear`. |
| `os_version_if_relevant`, `capture_app_version` | Actual version strings; OS may be null. |
| `lighting_id`, `lighting_type` | Stable setup ID and factual description. |
| `repeat_id` | Zero-based independent image repeat number. |
| `makeup_protocol` | `none`, `present`, or `unknown`; primary protocol requires `none`. |
| `distance` | Positive camera-to-subject distance in metres. |
| `pose_metadata_if_known` | Optional numeric dictionary; null when not measured. |
| `exposure_metadata_if_available`, `white_balance_metadata_if_available` | Optional numeric metadata dictionaries; preserve null when unavailable. |
| `reference_region` | `left_cheek` or `right_cheek`, anatomical side. |
| `ground_truth_L`, `ground_truth_a`, `ground_truth_b` | Mean of the valid stored instrument repeats, D65/2°. |
| `reference_illuminant`, `reference_observer` | Literals `D65` and `2`. |
| `reference_instrument` | Actual manufacturer/model and serial or stable serial alias. |
| `reference_measurement_id` | Unique ID for one subject/session/cheek repeat set. |
| `reference_geometry`, `reference_mode`, `reference_aperture_mm` | Actual locked instrument values; positive aperture. |
| `reference_calibration_id` | Links to the calibration event log. |
| `reference_timestamp` | Measurement timestamp including UTC offset. |
| `reference_repeats_lab` | At least two tuples; pilot protocol asks for at least three valid readings. |
| `face_bbox` | Pixel `(x, y, width, height)` in the canonical image after EXIF orientation and unmirroring. |
| `split` | Subject-level `train`, `validation`, `calibration`, or `test`. |
| `notes` | Factual deviations/exclusions; empty string if none. |

The target mean must match `reference_repeats_lab` within 0.05 per component. Reusing a `reference_measurement_id` with different content, an `image_id` with inconsistent capture metadata, or image bytes across subjects/splits is invalid.

## Auxiliary files

Supply a calibration-event table keyed by `reference_calibration_id`; capture/session timestamps; raw instrument export; invalid reading log with reasons; registration annotations linking ring/aperture location to each image; lighting setup sheets; device/app settings; operator IDs; consent status; and a data card based on `data_card_template.md`. Direct identity and contact linkage must remain outside this handoff.

Before delivery, run `uv run python scripts/validate_dataset.py --manifest <local-root>/manifest.jsonl` (use the script's current help if its option differs) and retain the exact command/output. Do not send participant files through source control or external model services.

Real-target training additionally requires `measurement_approval.json` in the data root with `schema_version: "1.0"`, `dataset_hash` equal to the manifest SHA256, `decision: "PASS"`, a real `reviewer_role`, `protocol_id`, positive preregistered `repeatability_p95_limit`, and `reference_accuracy_reviewed: true`, `registration_reviewed: true`. This is a signed-off research decision supplied after physical protocol review, not a file to fabricate now. The loader recomputes repeatability and blocks training if the threshold is exceeded. No such approval has been created for real data.
