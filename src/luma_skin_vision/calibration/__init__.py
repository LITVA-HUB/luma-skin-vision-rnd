"""Split calibration on subject maxima; no conditional selective-risk guarantee.

Finite-sample one-sided bound assumes exchangeable subject bundles and frozen
predictor, capture protocol and score. It does not guarantee safety under shift.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(eq=True)
class Calibrator:
    schema_version: str
    model_hash: str
    data_kind: str
    alpha: float
    tolerance: float
    residual_quantile: float | None
    subject_count: int
    domain_validated: bool = False

    def __post_init__(self):
        if (
            self.schema_version != "1.0"
            or not self.model_hash
            or self.data_kind not in ("SYNTHETIC", "INSTRUMENT", "PHOTO_REFERENCE")
        ):
            raise ValueError("invalid calibration provenance")
        if (
            not 0 < self.alpha < 1
            or not math.isfinite(self.tolerance)
            or self.tolerance <= 0
            or self.subject_count < 1
        ):
            raise ValueError("invalid calibration parameters")
        if self.residual_quantile is not None and not math.isfinite(self.residual_quantile):
            raise ValueError("nonfinite residual quantile must be encoded as null")
        if self.domain_validated:
            raise ValueError("No validated operating domain is supported in this research version")

    @classmethod
    def fit(
        cls,
        predicted_error,
        observed_error,
        subjects,
        *,
        split,
        model_hash,
        data_kind,
        alpha=0.1,
        tolerance=5,
    ):
        if split != "calibration":
            raise ValueError("calibration split required")
        p, y, s = np.asarray(predicted_error), np.asarray(observed_error), np.asarray(subjects)
        if p.ndim != 1 or p.shape != y.shape or len(p) != len(s) or not len(p):
            raise ValueError("invalid calibration vectors")
        if not np.isfinite(p).all() or not np.isfinite(y).all() or np.any(p < 0) or np.any(y < 0):
            raise ValueError("invalid calibration scores")
        if not 0 < alpha < 1 or tolerance <= 0:
            raise ValueError("invalid alpha or tolerance")
        scores = sorted(float((y - p)[s == k].max()) for k in np.unique(s))
        k = math.ceil((len(scores) + 1) * (1 - alpha))
        q = scores[k - 1] if k <= len(scores) else None
        return cls("1.0", model_hash, data_kind, alpha, tolerance, q, len(scores))

    def upper_error(self, predicted_error):
        p = np.asarray(predicted_error)
        if not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("invalid predicted error")
        return np.maximum(
            0, p + (self.residual_quantile if self.residual_quantile is not None else np.inf)
        )

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value["schema_version"] != "1.0":
            raise ValueError("unsupported calibration version")
        return cls(**value)
