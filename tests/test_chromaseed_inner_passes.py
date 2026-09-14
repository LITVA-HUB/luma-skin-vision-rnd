"""Guard development roles and retain every prefix, including harmful ones."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chromaseed_inner_passes import check_partition, summarize  # noqa: E402

from luma_skin_vision.color import delta_e00  # noqa: E402


def example():
    target = np.tile([50., 10., 15.], (5, 1))
    people = np.array(["a", "b", "b", "b", "b"])
    predictions = np.broadcast_to(target[None, :, None], (3, 5, 4, 3)).copy()
    predictions[:, :, :, 0] += [8, 5, 3, 1]
    predictions[:, 0, :, 0] += 7
    return predictions, target, people


def test_all_prefixes_and_equal_person_seed_weights_are_preserved():
    predictions, target, people = example()
    result = summarize(predictions, target, people)
    assert [p["passes"] for p in result["prefixes"]] == [1, 2, 3, 4]
    assert result["descriptive_best_prefix"] == 4
    assert result["adopted_policy"] is None
    for j, row in enumerate(result["prefixes"]):
        errors = delta_e00(predictions[:, :, j], target[None])
        expected = (errors[:, 0].mean() + errors[:, 1:].mean()) / 2
        assert row["person_mean"] == pytest.approx(expected)
        assert row["person_mean"] != pytest.approx(row["image_mean"])
    reverse = summarize(predictions[:, :, ::-1], target, people)
    assert reverse["descriptive_best_prefix"] == 1
    assert reverse["prefixes"][-1]["person_mean"] > reverse["prefixes"][0]["person_mean"]


def test_small_prediction_change_can_coexist_with_large_reference_error():
    predictions, target, people = example()
    predictions[..., 0] = [80, 80.01, 80.02, 80.03]
    result = summarize(predictions, target, people)
    row = next(r for r in result["plateaus"] if r["passes"] == 2 and r["threshold"] == 0.1)
    assert row["selected_seed_rows"] == 15
    assert row["current_person_weighted_error"] > 20
    assert row["later_minus_current_error"] > 0
    assert result["adopted_policy"] is None


def test_empty_plateau_has_no_invented_quality_value():
    predictions, target, people = example()
    result = summarize(predictions, target, people)
    row = next(r for r in result["plateaus"] if r["passes"] == 2 and r["threshold"] == 0.1)
    assert row["selected_seed_rows"] == 0
    assert row["current_person_weighted_error"] is None
    assert row["later_minus_current_error"] is None


def test_partition_uses_actual_gapped_indices_and_subject_exclusion():
    people = np.array(["a", "held", "a", "held", "b", "c"])
    check_partition(np.array([0, 2]), np.array([4, 5]), np.array([0, 2, 4, 5]), people)
    with pytest.raises(ValueError, match="person"):
        check_partition(np.array([0, 4]), np.array([2, 5]), np.array([0, 2, 4, 5]), people)
    with pytest.raises(ValueError):
        check_partition(np.array([0, 2]), np.array([1, 4, 5]), np.array([0, 2, 4, 5]), people)
    with pytest.raises(ValueError):
        check_partition(np.array([0, 2]), np.array([4, 4, 5]), np.array([0, 2, 4, 5]), people)


@pytest.mark.parametrize("bad", ["shape", "nonfinite", "missing_person"])
def test_bad_prediction_arrays_fail(bad):
    predictions, target, people = example()
    if bad == "shape":
        predictions = predictions[:, :, :3]
    elif bad == "nonfinite":
        predictions[0, 0, 0, 0] = np.nan
    else:
        people = people[:-1]
    with pytest.raises(ValueError):
        summarize(predictions, target, people)
