"""Synthetic-only selector separation and frozen evaluation integrity."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def api():
    path = Path(__file__).resolve().parents[1] / "scripts/cc_v2_statistics_eval.py"
    spec = importlib.util.spec_from_file_location("statistics_eval_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(api, tmp_path):
    data, run, out = tmp_path / "data", tmp_path / "run", tmp_path / "heads"
    data.mkdir()
    rows = []
    for part, count in [("train", 8), ("val", 2), ("risk", 25), ("cal", 6), ("test", 3)]:
        rows += [
            {
                "id": f"{part}{i}",
                "subset": part,
                "group": f"risk{i // 5}" if part == "risk" else part,
                "camera": "Canon EOS 550D",
            }
            for i in range(count)
        ]
    rng = np.random.default_rng(81)
    np.savez_compressed(
        data / "cube.npz",
        images=rng.uniform(0.1, 1, (len(rows), 3, 8, 8)),
        gt=rng.uniform(0.2, 1, (len(rows), 3)),
    )
    api.write_json(data / "cube_manifest.json", rows)
    api.statistics.screen(
        data, run, tmp_path / "screen.json", names=["ridge1"], expected_counts=(8, 2)
    )
    lock = tmp_path / "estimator_lock.json"
    api.write_json(
        lock,
        {
            "statistics_controls": ["gw_ridge1"],
            "statistics_source_screen_sha256": api.sha256(run / "screen.json"),
        },
    )
    return data, run, out, lock, rows


def test_source_only_fit_and_matched_grid(api, tmp_path, monkeypatch):
    data, run, out, lock, rows = fixture(api, tmp_path)
    reader, seen = api.statistics.read_npz_rows, []

    def tracked(path, key, selected, **kwargs):
        if key == "gt":
            seen.extend(selected.tolist())
            assert all(rows[i]["subset"] in ("risk", "cal") for i in selected)
        return reader(path, key, selected, **kwargs)

    monkeypatch.setattr(api.statistics, "read_npz_rows", tracked)
    state = api.fit(run, "gw_ridge1", data, out, lock, expected_counts=(25, 6))
    assert len(seen) == 31
    assert len(state["fit_ids"]) == 25 and len(state["cal_ids"]) == 6
    assert set(state["heads"]) == {"context", "cheap", "combined"}
    for head in state["heads"].values():
        assert [c["name"] for c in head["candidates"]] == list(api.GRID)
        assert len(head["cal_scores"]) == 6
        assert head["scale"] > 0
    for fold in state["folds"]:
        fit_groups = {rows[10 + i]["group"] for i in fold["train"]}
        val_groups = {rows[10 + i]["group"] for i in fold["validation"]}
        assert fit_groups.isdisjoint(val_groups)


def test_selection_lock_and_changed_inputs_rejected(api, tmp_path):
    data, run, out, lock, _ = fixture(api, tmp_path)
    with pytest.raises(ValueError, match="selection lock"):
        api.fit(run, "direct_ridge1", data, out, lock, expected_counts=(25, 6))
    api.fit(run, "gw_ridge1", data, out, lock, expected_counts=(25, 6))
    head_lock = tmp_path / "heads_lock.json"
    api.write_json(head_lock, {"statistics_selectors": {}})
    with pytest.raises(ValueError, match="head lock"):
        api.evaluate(out, head_lock, tmp_path / "result.json")
    api.write_json(
        head_lock, {"statistics_selectors": {"gw_ridge1": api.sha256(out / "selection.json")}}
    )
    with (data / "cube_manifest.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="binding"):
        api.evaluate(out, head_lock, tmp_path / "result.json")


def test_synthetic_external_invalid_refusal_and_bound_domains(api, tmp_path):
    data, run, out, lock, _ = fixture(api, tmp_path)
    api.fit(run, "gw_ridge1", data, out, lock, expected_counts=(25, 6))
    head_lock = tmp_path / "heads_lock.json"
    api.write_json(
        head_lock, {"statistics_selectors": {"gw_ridge1": api.sha256(out / "selection.json")}}
    )
    external, manifest = tmp_path / "external.npz", tmp_path / "external.json"
    np.savez_compressed(external, images=np.zeros((2, 3, 8, 8)), gt=np.ones((2, 3)))
    api.write_json(
        manifest, [{"id": str(i), "group": "fresh", "camera": "Synthetic"} for i in range(2)]
    )
    output = tmp_path / "result.json"
    api.evaluate(out, head_lock, output, external=(external, manifest))
    result = json.loads(output.read_text())
    assert set(result["domains"]) == {"source_regression", "fresh_all", "fresh_Synthetic"}
    for record in result["domains"]["fresh_all"].values():
        assert record["invalid_n"] == 2
        assert all(item["n"] == 0 for item in record["frozen_source_thresholds"].values())
