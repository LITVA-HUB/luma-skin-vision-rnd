# `tests/test_cc_v2_statistics.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v2_statistics.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Synthetic engineering checks for source-only conventional statistics baselines.

SHA-256 исходника: `9238a1c8b79f6b438f2f841e9c2ece3cc6bbebac226b45d81235bd64f246015d`. Строк: **195**.

## Зависимости

```python
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `module` | FunctionDef | См. реализацию | [L12](../../../../tests/test_cc_v2_statistics.py#L12) |
| `test_statistics_feature_order_capacity_and_diagonal_invariance` | FunctionDef | См. реализацию | [L21](../../../../tests/test_cc_v2_statistics.py#L21) |
| `test_residual_targets_and_reconstructed_predictions_transform_consistently` | FunctionDef | См. реализацию | [L44](../../../../tests/test_cc_v2_statistics.py#L44) |
| `make_data` | FunctionDef | См. реализацию | [L62](../../../../tests/test_cc_v2_statistics.py#L62) |
| `test_streaming_reader_deserializes_only_selected_source_rows` | FunctionDef | См. реализацию | [L84](../../../../tests/test_cc_v2_statistics.py#L84) |
| `test_source_screen_is_train_only_and_saves_all_candidates` | FunctionDef | См. реализацию | [L98](../../../../tests/test_cc_v2_statistics.py#L98) |
| `test_frozen_prediction_ignores_labels_and_checks_model_hash` | FunctionDef | См. реализацию | [L125](../../../../tests/test_cc_v2_statistics.py#L125) |
| `test_fitted_gw_ridge_preserves_diagonal_equivariance` | FunctionDef | См. реализацию | [L148](../../../../tests/test_cc_v2_statistics.py#L148) |
| `test_frozen_prediction_rejects_input_mutation_before_publication` | FunctionDef | См. реализацию | [L167](../../../../tests/test_cc_v2_statistics.py#L167) |

## Все тестовые определения (7)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_statistics_feature_order_capacity_and_diagonal_invariance` · L21

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_residual_targets_and_reconstructed_predictions_transform_consistently` · L44

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_streaming_reader_deserializes_only_selected_source_rows` · L84

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_source_screen_is_train_only_and_saves_all_candidates` · L98

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_frozen_prediction_ignores_labels_and_checks_model_hash` · L125

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_fitted_gw_ridge_preserves_diagonal_equivariance` · L148

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_frozen_prediction_rejects_input_mutation_before_publication` · L167

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('changed', ['npz', 'manifest'])
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
```

</details>
