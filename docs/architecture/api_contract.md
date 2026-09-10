# Local integration contract v1

`luma_skin_vision.api.analyze(run, image_path, bbox=None, detector=None)` is a local function. A supplied bbox is in canonical post-EXIF unmirrored coordinates. In a future production adapter an explicit mirroring/capture contract is required; ambiguous mirroring is unsupported. No FastAPI server or Luma code was modified.

Response fields: `schema_version`, `status`, `measurement`, `uncertainty`, `quality_flags`, `rejection_reason`, `model_version`, `preprocessing_version`, `undertone`, `profile_update`, `evidence_kind`. Current models return `UNSUPPORTED`, null measurement/uncertainty values, `undertone='unknown'`, `requires_user_confirmation=true`, `may_update=false`.

Future validated measurement: per-region continuous Lab, D65/2° reference, optional derived sRGB display swatch with gamut mapping declared. Error output must identify its target, units, calibration artifact and operating domain. A raw neural or detector score must not be called calibrated probability. A display swatch is not a physical pigment measurement or product shade recommendation.

| Status | Meaning | Product action |
|---|---|---|
| ACCEPT | Quality, calibrated error rule and documented supported domain all pass | Show result; ask confirmation before any profile change |
| RETAKE | Supported acquisition regime but photo/estimated measurement is unreliable | Suggest guided recapture; no measurement-driven profile update |
| UNSUPPORTED | Missing detector/calibration, unknown domain or current unvalidated model | No color claim; unsupported reason; no profile update |

Malformed image files raise a documented local `ValueError`; an HTTP adapter must map this to a client error without logging image bytes. Runtime no-domain status takes precedence over a retake suggestion. Invalid/NaN/infinite error bounds cannot be accepted. The isolated `selective.decide` function is tested for future integration, but current API never upgrades an artifact to a validated domain.

The supplied `2.0-proposed` rejection JSON in docs/context is historical proposal evidence, not this API's schema or a verified existing Luma endpoint.
