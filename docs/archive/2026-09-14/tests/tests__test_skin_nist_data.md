# `tests/test_skin_nist_data.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_nist_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `5908df440744857cd92715de03e55ee8ebdd1aa46ffb9553671436921e3dcb04`. Строк: **39**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fixture_text` | FunctionDef | См. реализацию | [L10](../../../../tests/test_skin_nist_data.py#L10) |
| `test_nist_triplicates_stay_with_the_same_subject` | FunctionDef | См. реализацию | [L15](../../../../tests/test_skin_nist_data.py#L15) |
| `test_nist_checks_average_and_grid_without_clipping` | FunctionDef | См. реализацию | [L24](../../../../tests/test_skin_nist_data.py#L24) |
| `test_nist_retains_author_repeat_labels_including_r4` | FunctionDef | См. реализацию | [L32](../../../../tests/test_skin_nist_data.py#L32) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_nist_triplicates_stay_with_the_same_subject` · L15

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_nist_triplicates_stay_with_the_same_subject():
    from skin_nist_data import parse_nist
    arrays = parse_nist(fixture_text())
    np.testing.assert_array_equal(arrays['wavelength_nm'], [250, 253])
    assert arrays['repeats'].shape == (1, 3, 2)
    np.testing.assert_allclose(arrays['average'], [[.2, .3]])
    assert arrays['partition'].shape == (1,)
```

</details>

### `test_nist_checks_average_and_grid_without_clipping` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_nist_checks_average_and_grid_without_clipping():
    from skin_nist_data import parse_nist
    with pytest.raises(ValueError, match='average'):
        parse_nist(fixture_text().replace('.3,.2', '.3,.9'))
    with pytest.raises(ValueError, match='wavelength'):
        parse_nist(fixture_text().replace('253,', '249,'))
```

</details>

### `test_nist_retains_author_repeat_labels_including_r4` · L32

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_nist_retains_author_repeat_labels_including_r4():
    from skin_nist_data import parse_nist
    arrays = parse_nist(fixture_text().replace('R1,R2,R3', 'R2,R3,R4'))
    np.testing.assert_array_equal(arrays['repeat_labels'], [['R2', 'R3', 'R4']])
    with pytest.raises(ValueError, match='layout'):
        parse_nist(fixture_text().replace('R1,R2,R3', 'R1,R2,R2'))
    with pytest.raises(ValueError, match='layout'):
        parse_nist(fixture_text().replace('R1,R2,R3', 'R1,R2,Other'))
```

</details>
