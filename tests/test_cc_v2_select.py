"""Synthetic-only selector integrity/leakage tests; no real target evaluation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from luma_skin_vision.cc.v2_experiment import SCHEMA, data_fingerprints, source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


@pytest.fixture(scope="module")
def selector():
    path = Path(__file__).resolve().parents[1] / "scripts/cc_v2_select.py"
    spec = importlib.util.spec_from_file_location("cc_v2_selector_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def bounded_cpu():
    with threadpool_limits(limits=2):
        yield


def write_predictions(run, name, npz, manifest, values):
    np.savez_compressed(run / name, **values)
    return {
        "sha256": sha256(run / name),
        "images": len(values["pred"]),
        "invalid_images": int((~values["valid"]).sum()),
        "input_npz": str(npz),
        "input_manifest": str(manifest),
        "input_hashes": {"npz": sha256(npz), "manifest": sha256(manifest)},
    }


def fixture_run(root, *, poison_heldout=False, invalid_test=False, perfect_cal=False):
    """Opaque checkpoint bytes are sufficient: selectors never instantiate a CNN."""
    data, run = root / "data", root / "run"
    data.mkdir(parents=True)
    run.mkdir()
    rows = []
    for part, count in [("train", 4), ("val", 2), ("risk", 20), ("cal", 4), ("test", 4)]:
        for i in range(count):
            rows.append(
                {
                    "id": f"{part}_{i}",
                    "subset": part,
                    "group": f"risk_{i // 4}" if part == "risk" else part,
                    "camera": "Canon EOS 550D",
                }
            )
    n = len(rows)
    rng = np.random.default_rng(184)
    gt = np.tile([1.0, 1.0, 1.0], (n, 1))
    pred = np.column_stack([np.linspace(0.5, 1.5, n), np.ones(n), np.linspace(1.2, 0.6, n)])
    pred /= np.linalg.norm(pred, axis=1, keepdims=True)
    values = {
        "pred": pred,
        "context": rng.normal(size=(n, 64)),
        "cheap_features": rng.normal(size=(n, 21)),
        "valid": np.ones(n, bool),
        "ids": np.array([r["id"] for r in rows]),
    }
    if poison_heldout:
        heldout = np.array([r["subset"] not in ("risk", "cal") for r in rows])
        gt[heldout] = np.nan
        values["context"][heldout] = np.nan
        values["cheap_features"][heldout] = np.nan
        values["pred"][heldout] = np.nan
    if invalid_test:
        values["valid"][-4:] = False
    if perfect_cal:
        cal = np.array([r["subset"] == "cal" for r in rows])
        values["pred"][cal] = gt[cal] / np.sqrt(3)
    np.savez_compressed(data / "cube.npz", gt=gt)
    write_json(data / "cube_manifest.json", rows)
    identity, snapshot = source_snapshot(run)
    config = {
        "schema_version": SCHEMA,
        "protocol": "official",
        "mode": "gw",
        "backbone": "small",
        "data_hashes": data_fingerprints(data),
        "source_snapshot": snapshot,
        **identity,
    }
    write_json(run / "config.json", config)
    (run / "model.pt").write_bytes(b"SYNTHETIC selector-only binding fixture; never torch.load")
    checkpoint = {
        "schema_version": SCHEMA,
        "checkpoint_file": "model.pt",
        "checkpoint_sha256": sha256(run / "model.pt"),
        "config_sha256": sha256(run / "config.json"),
        "source_hash": identity["source_hash"],
    }
    write_json(run / "checkpoint_manifest.json", checkpoint)
    output = write_predictions(
        run, "predictions.npz", data / "cube.npz", data / "cube_manifest.json", values
    )
    write_json(
        run / "predictions_manifest.json",
        {
            "schema_version": SCHEMA,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "config_sha256": checkpoint["config_sha256"],
            "training_source_hash": identity["source_hash"],
            "prediction_source_identity": identity,
            "outputs": {"predictions.npz": output},
            "fitting_or_error_evaluation": False,
        },
    )
    return run, data, rows


def alter_npz(path, key, change):
    with np.load(path, allow_pickle=False) as file:
        values = {name: file[name].copy() for name in file.files}
    values[key] = change(values[key])
    np.savez_compressed(path, **values)


def test_selector_fit_ignores_all_non_risk_cal_values_and_keeps_date_folds_disjoint(
    selector, tmp_path
):
    run, data, rows = fixture_run(tmp_path / "clean")
    poisoned, poisoned_data, _ = fixture_run(tmp_path / "poisoned", poison_heldout=True)
    selector.fit(run, data)
    selector.fit(poisoned, poisoned_data)
    normal = json.loads((run / "risk_v2/selection.json").read_text())
    other = json.loads((poisoned / "risk_v2/selection.json").read_text())
    mapping = {r["id"]: r for r in rows}
    assert set(normal["fit_ids"]).isdisjoint(normal["cal_ids"])
    assert len(normal["folds"]) == 5
    visited = []
    for fold in normal["folds"]:
        train = {mapping[normal["fit_ids"][i]]["group"] for i in fold["train"]}
        validation = {mapping[normal["fit_ids"][i]]["group"] for i in fold["validation"]}
        assert train.isdisjoint(validation)
        visited.extend(fold["validation"])
    assert sorted(visited) == list(range(len(normal["fit_ids"])))
    for block in ("context", "cheap", "combined"):
        a, b = normal["heads"][block], other["heads"][block]
        assert a["selected"] == b["selected"]
        assert a["scale"] == b["scale"]
        assert a["candidates"] == b["candidates"]
        assert {c["name"] for c in a["candidates"]} == {
            "ridge1",
            "ridge10",
            "ridge100",
            "hgb3",
            "hgb7",
        }


def test_fit_rejects_predictions_changed_after_model_export(selector, tmp_path):
    run, data, _ = fixture_run(tmp_path)
    alter_npz(run / "predictions.npz", "pred", lambda value: value[:, ::-1])
    with pytest.raises(ValueError, match="(?i)prediction|binding|changed"):
        selector.fit(run, data)


def test_fit_rejects_prediction_manifest_for_another_checkpoint(selector, tmp_path):
    run, data, _ = fixture_run(tmp_path)
    path = run / "predictions_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["checkpoint_sha256"] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError, match="(?i)checkpoint|binding|prediction"):
        selector.fit(run, data)


def test_fit_rejects_source_predictions_bound_to_a_different_input_cache(selector, tmp_path):
    run, data, rows = fixture_run(tmp_path)
    other = tmp_path / "other_input.npz"
    np.savez_compressed(other, gt=np.full((len(rows), 3), 2.0))
    path = run / "predictions_manifest.json"
    manifest = json.loads(path.read_text())
    source = manifest["outputs"]["predictions.npz"]
    source["input_npz"] = str(other)
    source["input_hashes"]["npz"] = sha256(other)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="(?i)input|dataset|binding|prediction"):
        selector.fit(run, data)


def test_frozen_policy_always_refuses_invalid_inputs_even_at_full_coverage(selector, tmp_path):
    run, data, _ = fixture_run(tmp_path, invalid_test=True)
    selector.fit(run, data)
    output = tmp_path / "evaluation.json"
    selector.evaluate(run, data, None, output)
    report = json.loads(output.read_text())["domains"]["source_regression"]
    for record in report.values():
        assert record["invalid_n"] == 4
        for frozen in record["frozen_source_thresholds"].values():
            assert frozen["n"] == 0
            assert frozen["coverage"] == 0
            assert frozen["mean_reproduction"] is None


def test_external_targets_must_match_prediction_export_binding(selector, tmp_path):
    run, data, _ = fixture_run(tmp_path)
    external = tmp_path / "external.npz"
    external_manifest = tmp_path / "external.json"
    rows = [{"id": f"e{i}", "camera": "synthetic", "group": f"e{i}"} for i in range(3)]
    write_json(external_manifest, rows)
    np.savez_compressed(external, gt=np.ones((3, 3)))
    values = {
        "pred": np.ones((3, 3)) / np.sqrt(3),
        "context": np.zeros((3, 64)),
        "cheap_features": np.zeros((3, 21)),
        "valid": np.ones(3, bool),
        "ids": np.array([r["id"] for r in rows]),
    }
    path = run / "predictions_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["outputs"]["external_predictions.npz"] = write_predictions(
        run, "external_predictions.npz", external, external_manifest, values
    )
    write_json(path, manifest)
    selector.fit(run, data)
    alter_npz(external, "gt", lambda value: value * np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="(?i)prediction|fingerprint|binding|changed|input"):
        selector.evaluate(
            run, data, [str(external), str(external_manifest)], tmp_path / "evaluation.json"
        )


def test_mixed_validity_caps_diagnostic_coverage_and_preserves_metric_units(selector):
    pred = np.column_stack([np.linspace(0.5, 1.5, 10), np.ones(10), np.ones(10)])
    gt = np.ones_like(pred)
    valid = np.array([False, False] + [True] * 8)
    scores = np.arange(10, dtype=float)
    ids = [f"image{i}" for i in range(10)]
    report = selector.selective_result(pred, gt, scores, ids, np.array([1e8, 1e9]), valid)
    errors = np.asarray(report["errors"])
    assert report["max_supported_coverage"] == 0.8
    assert report["selective"]["order"] == list(range(2, 10))
    assert report["selective"]["coverage"][-1] == 0.8
    assert report["selective"]["fixed"]["80"]["n"] == 8
    assert report["selective"]["fixed"]["80"]["mean"] == pytest.approx(errors[valid].mean())
    assert report["selective"]["fixed"]["60"]["n"] == 6
    assert report["selective"]["fixed"]["60"]["mean"] == pytest.approx(errors[2:8].mean())
    assert report["selective"]["fixed"]["90"]["attainable"] is False
    for frozen in report["frozen_source_thresholds"].values():
        assert frozen["n"] == 8
        assert frozen["coverage"] == 0.8


def test_zero_calibration_error_does_not_collapse_risk_ranking(selector, tmp_path):
    run, data, _ = fixture_run(tmp_path, perfect_cal=True)
    selector.fit(run, data)
    state = json.loads((run / "risk_v2/selection.json").read_text())
    for head in state["heads"].values():
        assert head["scale"] > 0
