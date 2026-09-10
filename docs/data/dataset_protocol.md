# Instrument-paired facial color collection protocol

Status: **PLANNED**. No participant measurements have been collected. Any synthetic records are engineering fixtures only.

## Measurement target

The primary target is the mean of repeated CIELAB measurements for a specified cheek site, reported under D65 and the CIE 2° standard observer. Left and right cheeks are separate targets and separate `Record` rows. Do not substitute a whole-face average, a skin-tone class, an image-derived Lab value, or an unregistered nearby measurement.

Use two physically defined sites:

- `left_cheek`: the center of the flat malar area on the participant's anatomical left, halfway along the line from the lateral alar crease to the preauricular point, then 15 mm inferior to that line.
- `right_cheek`: the mirrored site on the anatomical right.

At intake, place a skin-safe, removable registration ring around each site. The ring's inner opening must exceed the instrument aperture and expose untouched skin. Photograph the ring in a registration frame, record its center in image coordinates, then remove it for both instrument contact and production-like photographs. Do not draw on or cover the measured area. Record deviations such as facial hair, lesions, inability to seat the instrument, or site displacement; do not silently move the site.

## Locked instrument configuration

Before collecting pilot data, select and freeze one configuration in the protocol log. Record the actual instrument manufacturer/model, serial number, aperture in millimetres, measurement geometry, and reflectance mode (`SCI` or `SCE`) exactly as shown by the device or its documentation. These values have not been supplied and are therefore **DATA REQUIRED**, not defaults. Do not mix configurations within a session. A changed instrument, aperture, geometry, or mode starts a new protocol version and calibration identity.

Set the instrument to output or convert measurements to D65/2°. Preserve the native reading/export as immutable source data when available. `reference_instrument`, `reference_aperture_mm`, `reference_geometry`, and `reference_mode` must contain actual values, never `unknown`, guessed specifications, or synthetic labels for `INSTRUMENT` records.

## Participant and session preparation

Obtain documented consent before capture. Explain that face photographs and skin-color measurements are sensitive local research data, participation is voluntary, and withdrawal handling follows the study's approved consent text. Assign an opaque random `subject_id`; never encode names, contact details, dates of birth, ethnicity, appearance, or recruitment order in it. Keep the consent/contact linkage outside the dataset, access controlled and separately encrypted.

Require no facial makeup, tinted sunscreen, self-tanner, or recently applied colored product on the cheeks. Record `makeup_protocol: none` only after verbal confirmation and visual check; otherwise use `present` or `unknown` and exclude the sample from the primary no-makeup analysis. Ask participants not to rub or wash the sites immediately before measurement. Allow a documented acclimation period in the capture room and record its duration; choose the duration before the pilot and keep it fixed.

Create one `session_id` per subject visit. Record local timestamp with UTC offset for the session, every reference measurement, and every photograph. Record ambient conditions if instruments are available, without inventing missing values.

## Calibration and reference repeats

At the start of each session, perform the manufacturer's required white/black or tile calibration and assign a unique `reference_calibration_id`. Record timestamp, instrument serial number, tile identifier if applicable, outcome, and operator. Repeat calibration after any instrument warning, interruption, configuration change, or the manufacturer's specified interval. Never backfill calibration metadata.

For each cheek:

1. Confirm the registered physical site and clean instrument contact surface according to its approved procedure.
2. Acquire at least three consecutive readings, lifting and reseating the instrument between readings. Store every `(L*, a*, b*)` tuple in `reference_repeats_lab` in acquisition order.
3. If a reading is invalid or the probe slips, retain it in the raw log with a reason, reacquire, and exclude it only by a predeclared quality rule. Do not delete inconvenient readings.
4. Compute `ground_truth_L`, `ground_truth_a`, and `ground_truth_b` as the arithmetic mean of the valid stored repeats. The JSONL validator requires agreement within 0.05 due to rounding.
5. Use a unique `reference_measurement_id` for that subject, session, cheek, and repeat set. Record its timestamp with UTC offset.

Analyze within-site repeatability before model training. Report component differences and pairwise/within-site ΔE00 distributions overall and by operator/instrument where possible. The pilot proceeds to real-target training only if a pre-registered repeatability tolerance is met; set that tolerance after examining instrument specifications and before viewing model performance.

## Image capture matrix

The pilot target is 10–15 people, three distinct camera pipelines, four lighting conditions, and two independently initiated captures per subject × camera × light condition. Capture both cheeks in every usable image, yielding two region records that share `image_id` and image metadata but have separate reference fields.

Predeclare the three camera/device/module combinations and four reproducible lighting setups. Give each setup a stable `lighting_id` and document fixture type, placement, approximate subject/light geometry, and any measured illuminance/CCT. One condition should represent the intended capture domain; include controlled variation that is plausible for the product. Do not describe an uncontrolled condition more precisely than it was measured.

For every image, record device manufacturer/model, camera module, front/rear camera, OS if relevant, capture app/version, distance in metres, lighting ID/type, repeat ID, timestamps, and available exposure/white-balance metadata. Lock orientation, resolution policy, HDR/beautification/filter settings, and camera-to-subject pose where the app permits. Disable beauty filters. Preserve the original file bytes and compute SHA-256 before any processing.

The two image repeats must be separate captures: lower/re-raise or otherwise reset the device between them. Do not duplicate a file and label it as a repeat. Keep the face neutral, unobstructed, and approximately frontal while retaining normal capture variability.

## Region-to-image registration

Pair each instrument target with the same physical cheek location in the image. Retain a registration frame with the removable ring and a clean measurement frame whenever operationally feasible. Record ring center and diameter, image dimensions, transformation or landmark method used to transfer the site, annotator, timestamp, and a registration confidence/exception note in the auxiliary annotation file. The JSONL `face_bbox` uses pixel coordinates `(x, y, width, height)` in the canonical image after EXIF orientation and horizontal unmirroring when `image_mirrored` is true. `left_cheek` is anatomical left (image-right in this convention). Retain original bytes; normalization is performed when loading.

Registration must be checked independently on a sample before training. A generic anatomical cheek crop is not evidence that the pixels correspond to the instrument aperture. Images without defensible pairing can support engineering or photo-reference work but must not be labeled `INSTRUMENT` targets.

## Storage and quality control

Store images and manifests under an explicitly supplied local data root. Do not upload participant images to external services or commit them to Git. Restrict access, encrypt storage and backups using the project's approved local controls, and keep an access log. The repository does not prescribe a legal retention period; record the approved retention/deletion policy in the consent package and data card.

Run schema and file-integrity validation before analysis. Quarantine, rather than overwrite, records with missing images, checksum mismatch, duplicate content, inconsistent IDs, invalid bounding boxes, mixed target provenance, or subject leakage. Freeze a versioned manifest and record its hash for every experiment.

## Split policy

Assign subjects, never images, to mutually exclusive `train`, `validation`, `calibration`, and `test` splits. All sessions, cameras, lighting conditions, image repeats, and cheek rows for a subject stay in one split. Fit model weights and preprocessing parameters on train; choose architectures and thresholds on validation; fit uncertainty calibration and the final selective threshold on calibration; evaluate once on test.

In addition to the ordinary subject-held-out test, lock at least one device pipeline and one lighting condition before training as unseen-domain evaluations. No training, normalization, model selection, threshold selection, or calibration may use their outcomes. If the pilot is too small to support stable four-way plus domain-held-out inference, report uncertainty and use it to estimate collection scale rather than relaxing leakage controls.
