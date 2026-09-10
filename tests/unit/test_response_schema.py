import pytest

from luma_skin_vision.contracts import AnalysisResponse


def test_accept_cannot_be_empty_or_synthetic():
    with pytest.raises(ValueError):
        AnalysisResponse(
            schema_version="1.0",
            status="ACCEPT",
            measurement=None,
            evidence_kind="SYNTHETIC",
            model_version="smoke",
        )


def test_unsupported_contract_defaults_are_conservative():
    response = AnalysisResponse(model_version="smoke", evidence_kind="SYNTHETIC")
    assert response.measurement is None
    assert response.profile_update["may_update"] is False
