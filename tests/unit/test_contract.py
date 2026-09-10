import pytest

from luma_skin_vision.selective import decide


@pytest.mark.parametrize(
    "flags,domain,calibrated,upper,expected",
    [
        ([], False, True, 1, "UNSUPPORTED"),
        ([], True, False, 1, "UNSUPPORTED"),
        (["blur"], True, True, 1, "RETAKE"),
        ([], True, True, 7, "RETAKE"),
        ([], True, True, 3, "ACCEPT"),
        ([], True, True, float("nan"), "RETAKE"),
        ([], True, True, float("inf"), "RETAKE"),
        ([], True, True, -1, "RETAKE"),
    ],
)
def test_policy(flags, domain, calibrated, upper, expected):
    assert (
        decide(
            quality_flags=flags,
            domain_supported=domain,
            calibrated=calibrated,
            upper_error=upper,
            tolerance=5,
        )
        == expected
    )
