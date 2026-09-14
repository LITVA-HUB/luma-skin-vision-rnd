# `tests/test_cc_phone_benchmark.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_phone_benchmark.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `d1759a3d5ec89a8372d9de729e97f1abfc97b5895215a0e178e44a4ca0af6e0c`. Строк: **43**.

## Зависимости

```python
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../tests/test_cc_phone_benchmark.py#L9)

```python
SPEC = importlib.util.spec_from_file_location('phone_bench', Path(__file__).parents[1]/'scripts/cc_phone_benchmark.py')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_coverage_does_not_accept_refused_rows_or_hide_reference_exclusions` | FunctionDef | См. реализацию | [L14](../../../../tests/test_cc_phone_benchmark.py#L14) |
| `test_no_acceptable_prediction_is_explicitly_empty` | FunctionDef | См. реализацию | [L23](../../../../tests/test_cc_phone_benchmark.py#L23) |
| `test_data_hash_failure_precedes_numerical_file_read` | FunctionDef | См. реализацию | [L30](../../../../tests/test_cc_phone_benchmark.py#L30) |
| `test_primary_scene_bootstrap_preserves_phone_pairing` | FunctionDef | См. реализацию | [L39](../../../../tests/test_cc_phone_benchmark.py#L39) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_coverage_does_not_accept_refused_rows_or_hide_reference_exclusions` · L14

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_coverage_does_not_accept_refused_rows_or_hide_reference_exclusions():
    result = bench.risk_table(np.array([1., 3., 20.]), np.array([.2, .1, .0]),
                              np.array([True, True, False]), ['a', 'b', 'c'], planned=4)
    assert result['fixed']['100']['accepted'] == 2
    assert result['fixed']['100']['mean'] == 2
    assert result['fixed']['100']['coverage_scorable'] == 2/3
    assert result['fixed']['100']['coverage_planned'] == .5
```

</details>

### `test_no_acceptable_prediction_is_explicitly_empty` · L23

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_no_acceptable_prediction_is_explicitly_empty():
    result = bench.risk_table(np.array([50.]), np.array([1.]), np.array([False]), ['a'], 2)
    assert result['fixed']['80']['accepted'] == 0
    assert result['fixed']['80']['mean'] is None
    assert result['curve'] == []
```

</details>

### `test_data_hash_failure_precedes_numerical_file_read` · L30

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_data_hash_failure_precedes_numerical_file_read(tmp_path):
    path = tmp_path/'x.h5'
    path.write_bytes(b'bad')
    with pytest.raises(ValueError, match='digest'):
        bench.verified_path(tmp_path, 'x.h5', {'x.h5': {'sha256': '0'*64}})
    with pytest.raises(ValueError, match='outside'):
        bench.verified_path(tmp_path, '../escape', {})
```

</details>

### `test_primary_scene_bootstrap_preserves_phone_pairing` · L39

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_primary_scene_bootstrap_preserves_phone_pairing():
    draws = bench.scene_draws(['a','a','b','b'], repeats=5, seed=17)
    for ix in draws:
        assert list(ix).count(0) == list(ix).count(1)
        assert list(ix).count(2) == list(ix).count(3)
```

</details>
