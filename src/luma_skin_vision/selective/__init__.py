import math


def decide(*, quality_flags, domain_supported, calibrated, upper_error, tolerance):
    if not domain_supported or not calibrated:
        return "UNSUPPORTED"
    if (
        quality_flags
        or upper_error is None
        or not math.isfinite(upper_error)
        or upper_error < 0
        or not math.isfinite(tolerance)
        or tolerance <= 0
        or upper_error > tolerance
    ):
        return "RETAKE"
    return "ACCEPT"
