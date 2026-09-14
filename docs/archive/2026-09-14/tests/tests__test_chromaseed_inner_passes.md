# `tests/test_chromaseed_inner_passes.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_inner_passes.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Guard development roles and retain every prefix, including harmful ones.

SHA-256 исходника: `b4ebdffcbb7395c6185aeaadffd3fed463bfc791f2f3c5e2ab9a25ddf3846267`. Строк: **82**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_inner_passes import check_partition, summarize
from luma_skin_vision.color import delta_e00
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `example` | FunctionDef | См. реализацию | [L16](../../../../tests/test_chromaseed_inner_passes.py#L16) |
| `test_all_prefixes_and_equal_person_seed_weights_are_preserved` | FunctionDef | См. реализацию | [L25](../../../../tests/test_chromaseed_inner_passes.py#L25) |
| `test_small_prediction_change_can_coexist_with_large_reference_error` | FunctionDef | См. реализацию | [L41](../../../../tests/test_chromaseed_inner_passes.py#L41) |
| `test_empty_plateau_has_no_invented_quality_value` | FunctionDef | См. реализацию | [L52](../../../../tests/test_chromaseed_inner_passes.py#L52) |
| `test_partition_uses_actual_gapped_indices_and_subject_exclusion` | FunctionDef | См. реализацию | [L61](../../../../tests/test_chromaseed_inner_passes.py#L61) |
| `test_bad_prediction_arrays_fail` | FunctionDef | См. реализацию | [L73](../../../../tests/test_chromaseed_inner_passes.py#L73) |

## Все тестовые определения (5)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_all_prefixes_and_equal_person_seed_weights_are_preserved` · L25

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_small_prediction_change_can_coexist_with_large_reference_error` · L41

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_small_prediction_change_can_coexist_with_large_reference_error():
    predictions, target, people = example()
    predictions[..., 0] = [80, 80.01, 80.02, 80.03]
    result = summarize(predictions, target, people)
    row = next(r for r in result["plateaus"] if r["passes"] == 2 and r["threshold"] == 0.1)
    assert row["selected_seed_rows"] == 15
    assert row["current_person_weighted_error"] > 20
    assert row["later_minus_current_error"] > 0
    assert result["adopted_policy"] is None
```

</details>

### `test_empty_plateau_has_no_invented_quality_value` · L52

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_empty_plateau_has_no_invented_quality_value():
    predictions, target, people = example()
    result = summarize(predictions, target, people)
    row = next(r for r in result["plateaus"] if r["passes"] == 2 and r["threshold"] == 0.1)
    assert row["selected_seed_rows"] == 0
    assert row["current_person_weighted_error"] is None
    assert row["later_minus_current_error"] is None
```

</details>

### `test_partition_uses_actual_gapped_indices_and_subject_exclusion` · L61

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_partition_uses_actual_gapped_indices_and_subject_exclusion():
    people = np.array(["a", "held", "a", "held", "b", "c"])
    check_partition(np.array([0, 2]), np.array([4, 5]), np.array([0, 2, 4, 5]), people)
    with pytest.raises(ValueError, match="person"):
        check_partition(np.array([0, 4]), np.array([2, 5]), np.array([0, 2, 4, 5]), people)
    with pytest.raises(ValueError):
        check_partition(np.array([0, 2]), np.array([1, 4, 5]), np.array([0, 2, 4, 5]), people)
    with pytest.raises(ValueError):
        check_partition(np.array([0, 2]), np.array([4, 4, 5]), np.array([0, 2, 4, 5]), people)
```

</details>

### `test_bad_prediction_arrays_fail` · L73

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('bad', ['shape', 'nonfinite', 'missing_person'])
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
```

</details>
