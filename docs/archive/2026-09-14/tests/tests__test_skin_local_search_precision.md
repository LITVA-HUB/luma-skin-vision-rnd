# `tests/test_skin_local_search_precision.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_local_search_precision.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Checks for frozen post-training storage formats and their evaluation boundary.

SHA-256 исходника: `db95c0ac1fb6205c59fefb09e9232cdd9e65f648b16cc048b61ec3f6a62a796f`. Строк: **178**.

## Зависимости

```python
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 10](../../../../tests/test_skin_local_search_precision.py#L10)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `_model` | FunctionDef | См. реализацию | [L14](../../../../tests/test_skin_local_search_precision.py#L14) |
| `_populate_final_models` | FunctionDef | См. реализацию | [L50](../../../../tests/test_skin_local_search_precision.py#L50) |
| `test_storage_formats_and_int8_axis_scales_are_explicit` | FunctionDef | См. реализацию | [L66](../../../../tests/test_skin_local_search_precision.py#L66) |
| `test_rbf_int8_uses_beta_columns_centers_features_and_one_width_scale` | FunctionDef | См. реализацию | [L106](../../../../tests/test_skin_local_search_precision.py#L106) |
| `test_prepare_freezes_exactly_33_primary_models_without_outer_arrays` | FunctionDef | См. реализацию | [L118](../../../../tests/test_skin_local_search_precision.py#L118) |
| `test_evaluate_reports_every_format_without_selecting_one` | FunctionDef | См. реализацию | [L148](../../../../tests/test_skin_local_search_precision.py#L148) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_storage_formats_and_int8_axis_scales_are_explicit` · L66

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_storage_formats_and_int8_axis_scales_are_explicit():
    import skin_local_search_precision as precision

    source = _model("mlp")
    fp32 = precision.pack_model(source, "fp32_reference")
    fp16 = precision.pack_model(source, "fp16")
    int8 = precision.pack_model(source, "int8")

    for key, value in source.items():
        if value.dtype.kind == "f":
            assert fp32[key].dtype == np.float32
            assert fp16[key].dtype == np.float16
    assert precision.payload_stats(fp16) == {
        "numeric_scalars": 2749,
        "numeric_bytes": 5498,
        "int8_scalars": 0,
        "float16_scalars": 2749,
        "float32_scalars": 0,
        "quantization_scale_scalars": 0,
    }

    for key in ("x_mean", "x_std", "y_mean", "y_std"):
        assert int8[key].dtype == np.float32
    for key in ("hidden_w", "hidden_b", "out_w", "out_b", "skip_w"):
        assert int8[key].dtype == np.int8
        assert np.all(int8[f"{key}_scale"] > 0)
    assert int8["hidden_w_scale"].shape == (64,)
    assert int8["out_w_scale"].shape == (3,)
    assert int8["skip_w_scale"].shape == (3,)
    assert int8["hidden_b_scale"].shape == ()
    assert int8["out_b_scale"].shape == ()

    restored = precision.dequantize_model(int8)
    assert all(value.dtype == np.float32 for value in restored.values() if value.dtype.kind == "f")
    assert np.isfinite(restored["hidden_w"]).all()
    assert np.max(np.abs(restored["hidden_w"] - source["hidden_w"])) <= float(
        int8["hidden_w_scale"].max()
    ) / 2 + 1e-7
```

</details>

### `test_rbf_int8_uses_beta_columns_centers_features_and_one_width_scale` · L106

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_rbf_int8_uses_beta_columns_centers_features_and_one_width_scale():
    import skin_local_search_precision as precision

    packed = precision.pack_model(_model("guided_rbf"), "int8")
    assert packed["beta_scale"].shape == (3,)
    assert packed["centers_scale"].shape == (36,)
    assert packed["widths_scale"].shape == ()
    assert np.all(packed["beta_scale"] > 0)
    assert np.all(packed["centers_scale"] > 0)
    assert float(packed["widths_scale"]) > 0
```

</details>

### `test_prepare_freezes_exactly_33_primary_models_without_outer_arrays` · L118

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_prepare_freezes_exactly_33_primary_models_without_outer_arrays(tmp_path):
    import skin_local_search_precision as precision

    run = tmp_path / "run"
    _populate_final_models(run)
    outer_decoy = run / "mixed" / "final" / "ridge_s17_outer.npz"
    np.savez(outer_decoy, prediction=np.zeros((2, 3)), target=np.ones((2, 3)))

    manifest = precision.prepare(run)
    assert manifest["source_model_count"] == 33
    assert len(manifest["models"]) == 33
    assert all("_outer" not in entry["source_model"] for entry in manifest["models"])
    artifacts = list((run / "precision").glob("*/*/*/quantized.npz"))
    assert len(artifacts) == 99
    assert (run / "precision" / "manifest.sha256").read_text().strip() == hashlib.sha256(
        (run / "precision" / "manifest.json").read_bytes()
    ).hexdigest()
    assert precision.prepare(run) == manifest

    source = run / manifest["models"][0]["source_model"]
    with source.open("ab") as stream:
        stream.write(b"changed")
    try:
        precision.prepare(run)
    except ValueError as error:
        assert "source model hash" in str(error)
    else:
        raise AssertionError("a frozen source mutation must be rejected")
```

</details>

### `test_evaluate_reports_every_format_without_selecting_one` · L148

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_evaluate_reports_every_format_without_selecting_one(tmp_path, monkeypatch):
    import skin_local_search_precision as precision

    run = tmp_path / "run"
    _populate_final_models(run)
    rows = 6
    rng = np.random.default_rng(91)
    data = {
        "color": rng.normal(size=(rows, 36)).astype(np.float64),
        "target": rng.normal(size=(rows, 3)).astype(np.float64),
        "patient": np.array(["a", "a", "b", "b", "c", "c"]),
        "site": np.array(["x", "y", "x", "y", "x", "y"]),
        "device": np.array(["SLR", "SLR", "ipod", "ipod", "SLR", "ipod"]),
    }
    for protocol in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        folder = run / protocol
        np.savez(folder / "roles.npz", fit=np.array([1, 1, 1, 0, 0, 0], dtype=bool),
                 held=np.array([0, 0, 0, 1, 1, 1], dtype=bool), folds=np.array([0, 1, 2]))
    cache = tmp_path / "train.npz"
    np.savez(cache, **data)
    monkeypatch.setattr(precision, "CACHE_HASH", hashlib.sha256(cache.read_bytes()).hexdigest())

    precision.prepare(run)
    result = precision.evaluate(run, cache)
    assert result["formats"] == ["fp32_reference", "fp16", "int8"]
    assert result["format_was_selected_on_outer"] is False
    assert len(result["models"]) == 99
    assert {row["format"] for row in result["models"]} == set(result["formats"])
    assert all("person_delta_e" in row and "deviation_from_fp32" in row for row in result["models"])
    stored = json.loads((run / "precision" / "evaluation.json").read_text())
    assert stored == result
```

</details>
