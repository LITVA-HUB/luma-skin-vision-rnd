"""Synthetic engineering checks for source-only conventional statistics baselines."""

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/cc_v2_statistics.py"
    assert path.is_file(), "Required additive statistics baseline is missing"
    spec = importlib.util.spec_from_file_location("cc_v2_statistics_under_test", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_statistics_feature_order_capacity_and_diagonal_invariance():
    api = module()
    rng = np.random.default_rng(49)
    x = rng.uniform(0.1, 1, (3, 3, 16, 20))
    x[:, :, :3, :6] = 0
    gain = np.exp(rng.uniform(-1, 1, (3, 3)))
    direct, _, _ = api.featurize(x, "direct")
    residual, anchor, valid = api.featurize(x, "gw")
    changed, other_anchor, other_valid = api.featurize(x * gain[:, :, None, None], "gw")
    assert direct.shape == residual.shape == (3, 51)
    assert len(api.FEATURE_COLUMNS) == 51
    assert valid.all() and other_valid.all()
    np.testing.assert_allclose(changed, residual, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(other_anchor, anchor * gain, rtol=1e-10)
    constant = np.broadcast_to(np.array([1.0, 2.0, 4.0])[None, :, None, None], (1, 3, 8, 8))
    feature, _, _ = api.featurize(constant, "direct")
    np.testing.assert_allclose(
        feature[0, :36].reshape(12, 3), np.tile(np.log(np.array([1, 2, 4]) / (7 / 3)), (12, 1))
    )
    np.testing.assert_allclose(feature[0, 36:48], 1)
    np.testing.assert_allclose(feature[0, 48:], 0)


def test_residual_targets_and_reconstructed_predictions_transform_consistently():
    api = module()
    anchor = np.array([[0.3, 0.5, 0.2], [0.4, 0.4, 0.5]])
    gt = np.array([[0.4, 0.6, 0.2], [0.2, 0.6, 0.7]])
    gain = np.array([[1.5, 0.7, 2], [0.6, 1.3, 1.2]])
    for mode in ("direct", "gw"):
        target = api.target_ratios(gt, anchor, mode)
        prediction = api.decode_prediction(target, anchor, mode)
        np.testing.assert_allclose(prediction, gt / np.linalg.norm(gt, axis=1, keepdims=True))
    np.testing.assert_allclose(
        api.target_ratios(gt, anchor, "gw"), api.target_ratios(gt * gain, anchor * gain, "gw")
    )
    zero, zero_anchor, valid = api.featurize(np.zeros((2, 3, 8, 8)), "gw")
    assert np.isfinite(zero).all() and not valid.any()
    prediction = api.decode_prediction(np.zeros((2, 2)), zero_anchor, "gw")
    assert np.isfinite(prediction).all() and (prediction > 0).all()


def make_data(path):
    path.mkdir()
    rng = np.random.default_rng(9)
    rows = [
        {"id": str(i), "subset": part, "group": part, "camera": "Canon EOS 550D"}
        for i, part in enumerate(
            ["test", "train", "risk", "train", "val", "cal", "train", "val", "train"]
        )
    ]
    x = rng.uniform(0.1, 1, (len(rows), 3, 12, 12)).astype(np.float32)
    gt = rng.uniform(0.2, 0.9, (len(rows), 3))
    for i, row in enumerate(rows):
        if row["subset"] not in ("train", "val"):
            x[i] = np.nan
            gt[i] = np.nan
        if row["subset"] == "val":
            x[i] *= np.array([4, 1, 0.2])[:, None, None]
    np.savez_compressed(path / "cube.npz", images=x, gt=gt)
    (path / "cube_manifest.json").write_text(json.dumps(rows))
    return rows, x, gt


def test_streaming_reader_deserializes_only_selected_source_rows(tmp_path):
    api = module()
    _, x, gt = make_data(tmp_path / "data")
    selected = np.array([1, 3, 4, 6, 7, 8])
    np.testing.assert_array_equal(
        api.read_npz_rows(tmp_path / "data/cube.npz", "images", selected), x[selected]
    )
    np.testing.assert_array_equal(
        api.read_npz_rows(tmp_path / "data/cube.npz", "gt", selected), gt[selected]
    )
    with pytest.raises(ValueError):
        api.read_npz_rows(tmp_path / "data/cube.npz", "images", [3, 1])


def test_source_screen_is_train_only_and_saves_all_candidates(tmp_path):
    api = module()
    data, out, summary = tmp_path / "data", tmp_path / "run", tmp_path / "summary.json"
    rows, x, _ = make_data(data)
    report = api.screen(data, out, summary, names=["ridge1"], expected_counts=(4, 2), threads=2)
    assert report["scope"] == "SOURCE TRAIN/VALIDATION ONLY"
    assert len(report["candidates"]) == 2
    assert len(report["source_ids"]["train"]) == 4
    assert len(report["source_ids"]["val"]) == 2
    assert all(
        row["id"] not in report["source_ids"]["train"] + report["source_ids"]["val"]
        for row in rows
        if row["subset"] not in ("train", "val")
    )
    assert all(np.isfinite(row["val_mean_reproduction"]) for row in report["candidates"])
    payload = joblib.load(out / "direct_ridge1.joblib")
    train = np.array([i for i, row in enumerate(rows) if row["subset"] == "train"])
    feature, _, _ = api.featurize(x[train], "direct")
    np.testing.assert_allclose(
        payload["model"].named_steps["standardscaler"].mean_, feature.mean(axis=0)
    )
    assert summary.is_file()
    assert all((out / row["model_file"]).is_file() for row in report["candidates"])
    with pytest.raises(FileExistsError):
        api.screen(data, out, summary, names=["ridge1"], expected_counts=(4, 2), threads=2)


def test_frozen_prediction_ignores_labels_and_checks_model_hash(tmp_path):
    api = module()
    data, out = tmp_path / "data", tmp_path / "run"
    make_data(data)
    api.screen(
        data, out, tmp_path / "summary.json", names=["ridge1"], expected_counts=(4, 2), threads=2
    )
    external, manifest = tmp_path / "external.npz", tmp_path / "external.json"
    np.savez_compressed(external, images=np.zeros((2, 3, 8, 8)), gt=np.full((2, 3), np.nan))
    manifest.write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
    output = tmp_path / "predictions.npz"
    api.predict_frozen(out, "gw_ridge1", external, manifest, output, threads=2)
    with np.load(output) as predictions:
        assert np.isfinite(predictions["pred"]).all()
        assert not predictions["valid"].any()
        assert predictions["context"].shape == (2, 51)
        assert predictions["cheap_features"].shape == (2, 53)
    with (out / "gw_ridge1.joblib").open("ab") as file:
        file.write(b"modified")
    with pytest.raises(ValueError, match="(?i)model|artifact|hash"):
        api.predict_frozen(out, "gw_ridge1", external, manifest, tmp_path / "other.npz", threads=2)


def test_fitted_gw_ridge_preserves_diagonal_equivariance():
    api = module()
    rng = np.random.default_rng(2026)
    images = rng.lognormal(0, 0.8, (100, 3, 16, 20))
    features, anchors, _ = api.featurize(images, "gw")
    target = rng.normal(0, 0.2, (100, 2))
    model = api.estimator("ridge1").fit(features, target)
    payload = {"model": model, "mode": "gw"}
    gain = rng.uniform(0.25, 4, (100, 3))
    before = api.predict_arrays(payload, images)["pred"]
    after = api.predict_arrays(payload, images * gain[:, :, None, None])["pred"]
    expected = before * gain
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(after, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(features[:, :3], np.zeros((100, 3)))
    np.testing.assert_array_equal(model.named_steps["standardscaler"].scale_[:3], np.ones(3))


@pytest.mark.parametrize("changed", ["npz", "manifest"])
def test_frozen_prediction_rejects_input_mutation_before_publication(
    tmp_path, monkeypatch, changed
):
    api = module()
    data, out = tmp_path / "data", tmp_path / "run"
    make_data(data)
    api.screen(data, out, tmp_path / "summary.json", names=["ridge1"], expected_counts=(4, 2))
    source, manifest, output = (
        tmp_path / "input.npz",
        tmp_path / "input.json",
        tmp_path / "pred.npz",
    )
    np.savez_compressed(source, images=np.ones((2, 3, 8, 8)))
    manifest.write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
    original = api.predict_arrays

    def mutate(payload, images):
        values = original(payload, images)
        if changed == "npz":
            np.savez_compressed(source, images=images * np.array([1, 2, 3])[None, :, None, None])
        else:
            manifest.write_text(json.dumps([{"id": "changed"}, {"id": "b"}]))
        return values

    monkeypatch.setattr(api, "predict_arrays", mutate)
    with pytest.raises(ValueError, match="(?i)input.*changed"):
        api.predict_frozen(out, "gw_ridge1", source, manifest, output)
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
