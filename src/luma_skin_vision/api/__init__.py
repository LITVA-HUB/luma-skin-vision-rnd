"""Conservative research API; no current artifact is approved to update profiles."""

from pathlib import Path

from luma_skin_vision.preprocessing import decode_image, quality_gate
from luma_skin_vision.training import load_run


def analyze(run, image_path, *, bbox=None, detector=None):
    _, metadata = load_run(run)
    response = {
        "schema_version": "1.0",
        "status": "UNSUPPORTED",
        "measurement": None,
        "uncertainty": {
            "predicted_delta_e00": None,
            "probability_within_tolerance": None,
            "upper_error": None,
            "calibration_id": None,
        },
        "quality_flags": [],
        "rejection_reason": "no_validated_operating_domain",
        "model_version": Path(run).name,
        "preprocessing_version": "srgb-exif-v1",
        "undertone": "unknown",
        "profile_update": {"requires_user_confirmation": True, "may_update": False},
        "evidence_kind": metadata["data_kind"],
    }
    rgb = decode_image(image_path)
    if bbox is None and detector is not None:
        bbox = detector.detect(rgb)["bbox"]
    if bbox is None:
        response["rejection_reason"] = "face_localization_unavailable"
    else:
        response["quality_flags"] = quality_gate(rgb, bbox)
    return response
