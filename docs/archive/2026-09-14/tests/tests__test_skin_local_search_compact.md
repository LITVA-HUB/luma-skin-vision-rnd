# `tests/test_skin_local_search_compact.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_local_search_compact.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent synthetic checks for the frozen-prefix compression study.

SHA-256 исходника: `7f1895cb627173c04d251aa00a6113da4a3cc02a6bccf399acc08ff42c7d283e`. Строк: **146**.

## Зависимости

```python
import json
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `_weights` | FunctionDef | См. реализацию | [L13](../../../../tests/test_skin_local_search_compact.py#L13) |
| `_independent_design` | FunctionDef | См. реализацию | [L23](../../../../tests/test_skin_local_search_compact.py#L23) |
| `test_refit_prefix_preserves_order_and_matches_independent_weighted_ridge` | FunctionDef | См. реализацию | [L32](../../../../tests/test_skin_local_search_compact.py#L32) |
| `test_refit_prefix_rejects_non_rbf_or_invalid_prefix_without_mutation` | FunctionDef | См. реализацию | [L88](../../../../tests/test_skin_local_search_compact.py#L88) |
| `test_prefix_prediction_rejects_inconsistent_payload` | FunctionDef | См. реализацию | [L114](../../../../tests/test_skin_local_search_compact.py#L114) |
| `test_evaluation_receipt_does_not_mutate_frozen_prefix_manifest` | FunctionDef | См. реализацию | [L131](../../../../tests/test_skin_local_search_compact.py#L131) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_refit_prefix_preserves_order_and_matches_independent_weighted_ridge` · L32

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_refit_prefix_preserves_order_and_matches_independent_weighted_ridge():
    from skin_local_search_compact import prefix_predict, refit_prefix

    rng = np.random.default_rng(611)
    x = rng.normal(size=(18, 4))
    y = rng.normal(size=(18, 3))
    person = np.repeat(np.arange(6), 3)
    site = np.array([f"p{p}s{s}" for p in range(6) for s in (0, 0, 1)])
    source = {
        "method": np.asarray("guided_rbf"),
        "x_mean": x.mean(0).astype(np.float32),
        "x_std": np.maximum(x.std(0), 1e-6).astype(np.float32),
        "y_mean": y.mean(0).astype(np.float32),
        "y_std": np.maximum(y.std(0), 1e-6).astype(np.float32),
        "centers": rng.normal(size=(64, 4)).astype(np.float32),
        "widths": rng.uniform(0.4, 1.5, size=64).astype(np.float32),
        "beta": rng.normal(size=(69, 3)).astype(np.float32),
    }
    original_centers = source["centers"].copy()

    model, receipt = refit_prefix(source, x, y, person, site, alpha=0.6, count=3)
    assert str(model["method"]) == "guided_rbf"
    np.testing.assert_array_equal(model["centers"], original_centers[:3])
    np.testing.assert_array_equal(model["widths"], source["widths"][:3])
    np.testing.assert_array_equal(source["centers"], original_centers)
    assert model["beta"].shape == (1 + x.shape[1] + 3, 3)

    design = _independent_design(model, x)
    target = (y - model["y_mean"]) / model["y_std"]
    weights = _weights(person, site)
    penalty = np.eye(design.shape[1]) * 0.6
    penalty[0, 0] = 0.0
    expected = np.linalg.solve(
        design.T @ (weights[:, None] * design) + penalty,
        design.T @ (weights[:, None] * target),
    )
    np.testing.assert_allclose(model["beta"], expected, rtol=2e-6, atol=2e-6)

    query = rng.normal(size=(5, 4))
    expected_prediction = (
        _independent_design(model, query) @ model["beta"]
    ) * model["y_std"] + model["y_mean"]
    np.testing.assert_allclose(
        prefix_predict(model, query), expected_prediction, rtol=2e-6, atol=2e-6
    )
    assert receipt["prefix_atoms"] == 3
    assert receipt["source_atoms"] == 64
    assert receipt["numeric_scalars"] == sum(
        value.size for value in model.values() if value.dtype.kind == "f"
    )
    assert receipt["numeric_bytes"] == sum(
        value.nbytes for value in model.values() if value.dtype.kind == "f"
    )
    assert receipt["prefix_refit_seconds"] >= 0
```

</details>

### `test_refit_prefix_rejects_non_rbf_or_invalid_prefix_without_mutation` · L88

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_refit_prefix_rejects_non_rbf_or_invalid_prefix_without_mutation():
    from skin_local_search_compact import refit_prefix

    x = np.arange(24, dtype=np.float64).reshape(6, 4)
    y = np.arange(18, dtype=np.float64).reshape(6, 3)
    person = np.repeat(np.arange(3), 2)
    site = np.arange(6)
    source = {
        "method": np.asarray("ridge"),
        "x_mean": np.zeros(4, dtype=np.float32),
        "x_std": np.ones(4, dtype=np.float32),
        "y_mean": np.zeros(3, dtype=np.float32),
        "y_std": np.ones(3, dtype=np.float32),
        "centers": np.zeros((64, 4), dtype=np.float32),
        "widths": np.ones(64, dtype=np.float32),
        "beta": np.zeros((69, 3), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="RBF"):
        refit_prefix(source, x, y, person, site, 1.0, 3)
    source["method"] = np.asarray("random_rbf")
    with pytest.raises(ValueError, match="count"):
        refit_prefix(source, x, y, person, site, 1.0, 0)
    with pytest.raises(ValueError, match="count"):
        refit_prefix(source, x, y, person, site, 1.0, 64)
```

</details>

### `test_prefix_prediction_rejects_inconsistent_payload` · L114

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_prefix_prediction_rejects_inconsistent_payload():
    from skin_local_search_compact import prefix_predict

    model = {
        "method": np.asarray("random_rbf"),
        "x_mean": np.zeros(2, dtype=np.float32),
        "x_std": np.ones(2, dtype=np.float32),
        "y_mean": np.zeros(3, dtype=np.float32),
        "y_std": np.ones(3, dtype=np.float32),
        "centers": np.zeros((2, 2), dtype=np.float32),
        "widths": np.ones(2, dtype=np.float32),
        "beta": np.zeros((4, 3), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="beta"):
        prefix_predict(model, np.zeros((1, 2)))
```

</details>

### `test_evaluation_receipt_does_not_mutate_frozen_prefix_manifest` · L131

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_evaluation_receipt_does_not_mutate_frozen_prefix_manifest(tmp_path):
    from skin_local_search_compact import write_evaluation_receipts

    frozen = tmp_path / "frozen_prefixes.json"
    original = b'{"choices":{"guided_rbf":8},"outer_evaluation_run":false}\n'
    frozen.write_bytes(original)
    result = {"scope": "synthetic test", "models": [{"atoms": 8}]}

    write_evaluation_receipts(tmp_path, frozen, result)

    assert frozen.read_bytes() == original
    saved = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "evaluation_status.json").read_text(encoding="utf-8"))
    assert saved["models"] == result["models"]
    assert saved["frozen_prefixes_sha256"] == status["frozen_prefixes_sha256"]
    assert status["outer_evaluation_run"] is True
```

</details>
