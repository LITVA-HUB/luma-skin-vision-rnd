import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cc_v3_data import remainder


def sample(i):
    return {
        "image": {"path": f"camera/field_1_cameras/{i}.tiff"},
        "gt": {"path": f"camera/field_1_cameras/{i}.wp"},
        "selection_sha256": f"{i:064x}",
        "combined_record_range": {"bytes": 100},
    }


def test_remainder_is_exact_disjoint_metadata_population():
    result = remainder([sample(i) for i in range(256)], [sample(i) for i in range(128)])
    assert len(result) == 128
    assert [s["image"]["path"] for s in result] == [
        sample(i)["image"]["path"] for i in range(128, 256)
    ]


@pytest.mark.parametrize("kind", ["duplicate", "missing", "reordered_metadata"])
def test_remainder_refuses_ambiguous_or_changed_historical_members(kind):
    all_rows, old = [sample(i) for i in range(256)], [sample(i) for i in range(128)]
    if kind == "duplicate":
        all_rows[-1] = all_rows[-2]
    elif kind == "missing":
        old[-1] = sample(999)
    else:
        old[-1]["gt"]["path"] = "different.wp"
    with pytest.raises(ValueError):
        remainder(all_rows, old)
