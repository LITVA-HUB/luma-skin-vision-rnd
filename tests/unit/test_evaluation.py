import numpy as np
import pytest

from luma_skin_vision.calibration import Calibrator
from luma_skin_vision.evaluation import paired_bootstrap, repeatability, risk_coverage, summarize


def test_equal_coverage_and_ties():
    curve = risk_coverage([4, 1, 3, 2], [4, 1, 3, 2], ["d", "a", "c", "b"], coverages=[0.5, 1.0])
    assert curve[0]["accepted"] == 2 and curve[0]["mean_delta_e00"] == 1.5
    assert curve[1]["mean_delta_e00"] == 2.5
    tied = risk_coverage([1, 9], [0, 0], ["a", "b"], coverages=[0.5])
    reverse = risk_coverage([9, 1], [0, 0], ["b", "a"], coverages=[0.5])
    assert tied == reverse  # Tie break must not inspect true error or input row order.
    with pytest.raises(ValueError):
        risk_coverage([], [], [])


def test_paired_cluster_bootstrap():
    result = paired_bootstrap([4, 6, 8, 10], [3, 5, 7, 9], ["a", "a", "b", "b"], seed=1, draws=100)
    assert result["mean_difference"] == 1
    assert result["ci95"] == [1, 1]
    assert result["subjects"] == 2
    assert result == paired_bootstrap(
        [4, 6, 8, 10], [3, 5, 7, 9], ["a", "a", "b", "b"], seed=1, draws=100
    )


def test_summary_and_repeatability():
    y = np.array([[50, 0, 0], [50, 0, 0]])
    metrics = summarize(y, y, ["a", "b"])
    assert metrics["mean_delta_e00"] == 0 and metrics["catastrophic_rate"] == 0
    assert metrics["mae_lab"] == [0, 0, 0]
    assert (
        repeatability([[[50, 0, 0], [50, 0, 0]], [[40, 0, 0], [40, 0, 0]]])["median_pair_delta_e00"]
        == 0
    )
    assert np.isfinite(summarize(y, y, [1, 2])["between_subject_prediction_std_lab"]).all()


def test_calibration_split_quantile_serialization(tmp_path):
    c = Calibrator.fit(
        np.arange(20),
        np.arange(20) + 2,
        [f"s{i}" for i in range(20)],
        split="calibration",
        alpha=0.1,
        tolerance=5,
        model_hash="abc",
        data_kind="SYNTHETIC",
    )
    assert c.upper_error([1]).item() == 3
    c.save(tmp_path / "c.json")
    loaded = Calibrator.load(tmp_path / "c.json")
    assert loaded == c
    assert c.domain_validated is False
    with pytest.raises(ValueError, match="calibration"):
        Calibrator.fit([1], [1], ["a"], split="test", model_hash="a", data_kind="SYNTHETIC")
    # Finite sample order exceeds n: no finite guarantee available.
    small = Calibrator.fit(
        [1], [2], ["a"], split="calibration", alpha=0.1, model_hash="a", data_kind="SYNTHETIC"
    )
    assert np.isinf(small.upper_error([1])).all()


def test_invalid_calibrator_rejected(tmp_path):
    import json

    p = tmp_path / "bad.json"
    p.write_text(
        json.dumps(
            dict(
                schema_version="1.0",
                model_hash="x",
                data_kind="SYNTHETIC",
                alpha=2,
                tolerance=-1,
                residual_quantile=0,
                subject_count=10,
            )
        )
    )
    with pytest.raises(ValueError):
        Calibrator.load(p)
