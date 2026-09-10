from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    left_cheek_lab: tuple[float, float, float]
    right_cheek_lab: tuple[float, float, float]
    reference_illuminant: Literal["D65"] = "D65"
    reference_observer: Literal["2"] = "2"


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ACCEPT", "RETAKE", "UNSUPPORTED"] = "UNSUPPORTED"
    measurement: Measurement | None = None
    uncertainty: dict[str, float | str | None] = Field(
        default_factory=lambda: {
            "predicted_delta_e00": None,
            "probability_within_tolerance": None,
            "upper_error": None,
            "calibration_id": None,
        }
    )
    quality_flags: list[str] = Field(default_factory=list)
    rejection_reason: str | None = "no_validated_operating_domain"
    model_version: str
    preprocessing_version: Literal["srgb-exif-v1"] = "srgb-exif-v1"
    undertone: Literal["unknown"] = "unknown"
    profile_update: dict[str, bool] = Field(
        default_factory=lambda: {"requires_user_confirmation": True, "may_update": False}
    )
    evidence_kind: Literal["SYNTHETIC", "INSTRUMENT", "PHOTO_REFERENCE"]

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "ACCEPT":
            if (
                self.measurement is None
                or self.evidence_kind != "INSTRUMENT"
                or self.rejection_reason is not None
            ):
                raise ValueError(
                    "ACCEPT requires instrument-validated measurement and no rejection reason"
                )
            if not self.uncertainty.get("calibration_id"):
                raise ValueError("ACCEPT requires calibration provenance")
        elif self.measurement is not None or self.profile_update.get("may_update"):
            raise ValueError(
                "Rejected/unsupported analysis must not carry measurement or allow profile update"
            )
        if self.profile_update.get("requires_user_confirmation") is not True:
            raise ValueError("Explicit user confirmation is mandatory")
        return self
