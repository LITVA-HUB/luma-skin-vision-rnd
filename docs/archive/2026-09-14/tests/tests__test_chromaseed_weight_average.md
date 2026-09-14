# `tests/test_chromaseed_weight_average.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_weight_average.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Behavioral contract for compact same-trajectory weight means.

SHA-256 исходника: `4e553738a12d0b0301c2f9fc2db547e98878fdfd0d5023b246f6f5e58f850a9b`. Строк: **104**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_neural_prefix_numpy import predict
from chromaseed_weight_average import average, choose, recipes
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `model` | FunctionDef | См. реализацию | [L14](../../../../tests/test_chromaseed_weight_average.py#L14) |
| `ids` | FunctionDef | См. реализацию | [L31](../../../../tests/test_chromaseed_weight_average.py#L31) |
| `test_singleton_copy_does_not_alias_or_change_parent` | FunctionDef | См. реализацию | [L35](../../../../tests/test_chromaseed_weight_average.py#L35) |
| `test_arithmetic_changes_only_weights_with_one_fp32_cast` | FunctionDef | См. реализацию | [L45](../../../../tests/test_chromaseed_weight_average.py#L45) |
| `test_mixed_provenance_is_rejected` | FunctionDef | См. реализацию | [L57](../../../../tests/test_chromaseed_weight_average.py#L57) |
| `test_incompatible_normalizers_and_duplicate_steps_rejected` | FunctionDef | См. реализацию | [L64](../../../../tests/test_chromaseed_weight_average.py#L64) |
| `test_weight_mean_is_not_prediction_ensemble` | FunctionDef | См. реализацию | [L75](../../../../tests/test_chromaseed_weight_average.py#L75) |
| `test_recipe_sets_have_no_repeated_components_and_include_controls` | FunctionDef | См. реализацию | [L87](../../../../tests/test_chromaseed_weight_average.py#L87) |
| `test_selector_prefers_lower_error_then_shorter_cost_and_keeps_baseline` | FunctionDef | См. реализацию | [L98](../../../../tests/test_chromaseed_weight_average.py#L98) |

## Все тестовые определения (7)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_singleton_copy_does_not_alias_or_change_parent` · L35

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_singleton_copy_does_not_alias_or_change_parent():
    m = model(2)
    a = average([m], ids([0]))
    for k in m:
        np.testing.assert_array_equal(a[k], m[k])
        assert not np.shares_memory(a[k], m[k])
    a["c0"][0] = 9
    assert m["c0"][0] == 2
```

</details>

### `test_arithmetic_changes_only_weights_with_one_fp32_cast` · L45

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_arithmetic_changes_only_weights_with_one_fp32_cast():
    mm = [model(v) for v in (1, 2, 4)]
    a = average(mm, ids([512, 2048, 8192]))
    np.testing.assert_array_equal(a["c0"], np.full(3, 7 / 3, np.float32))
    for k in ("family", "original_k", "prefix", "x_mean", "x_std", "y_mean", "y_std"):
        np.testing.assert_array_equal(a[k], mm[0][k])
    assert sum(v.nbytes for v in a.values() if v.dtype.kind in "biufc") == 2886
```

</details>

### `test_mixed_provenance_is_rejected` · L57

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('field,value', [('context', 'mixed/fold1'), ('seed', 29), ('variants', 16), ('lr', 0.0003)])
def test_mixed_provenance_is_rejected(field, value):
    identity = ids([512, 2048])
    identity[1][field] = value
    with pytest.raises(ValueError):
        average([model(), model()], identity)
```

</details>

### `test_incompatible_normalizers_and_duplicate_steps_rejected` · L64

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_incompatible_normalizers_and_duplicate_steps_rejected():
    a, b = model(), model()
    b["x_mean"][0] = 1
    with pytest.raises(ValueError):
        average([a, b], ids([512, 2048]))
    with pytest.raises(ValueError):
        average([a, a], ids([512, 512]))
    with pytest.raises(ValueError):
        average([a, a], ids([2048, 512]))
```

</details>

### `test_weight_mean_is_not_prediction_ensemble` · L75

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_weight_mean_is_not_prediction_ensemble():
    a, b = model(), model()
    a["w0"][0, 0] = 1
    b["w0"][0, 0] = -1
    a["v0"][0, 0] = b["v0"][0, 0] = 1
    x = np.zeros((1, 36), np.float32)
    x[0, 0] = 1
    avg = average([a, b], ids([512, 2048]))
    assert predict(avg, x)[0, 0] == 0
    assert ((predict(a, x) + predict(b, x)) / 2)[0, 0] == 0.5
```

</details>

### `test_recipe_sets_have_no_repeated_components_and_include_controls` · L87

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_recipe_sets_have_no_repeated_components_and_include_controls():
    rr = recipes()
    assert len(rr) == 91 and len({r["name"] for r in rr}) == 91
    assert len({(r["variants"], r["lr"], tuple(r["steps"])) for r in rr}) == 91
    assert {
        m: sum(r["method"] == m for r in rr) for m in ("last", "pair", "prefix", "tail3")
    } == dict(last=31, pair=30, prefix=18, tail3=12)
    assert any(r["steps"] == [0, 512] for r in rr)
    assert any(r["steps"] == [8192, 32768, 131072] for r in rr)
```

</details>

### `test_selector_prefers_lower_error_then_shorter_cost_and_keeps_baseline` · L98

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selector_prefers_lower_error_then_shorter_cost_and_keeps_baseline():
    rr = recipes()
    baseline = {**rr[0], "clean": 4.0, "p90": 8.0}
    avg = {**next(r for r in rr if r["method"] == "pair"), "clean": 4.0, "p90": 8.0}
    assert choose([avg, baseline])["name"] == baseline["name"]
    avg["clean"] = 3.9
    assert choose([avg, baseline])["name"] == avg["name"]
```

</details>
