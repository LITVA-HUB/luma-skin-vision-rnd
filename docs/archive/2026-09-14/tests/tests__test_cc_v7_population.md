# `tests/test_cc_v7_population.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v7_population.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `84857a52c148c0e9751b9c6802a9daea540c79267edede1c4a3400f06f65397c`. Строк: **23**.

## Зависимости

```python
import sys
from pathlib import Path
import pytest
from cc_v7_external_population import select
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `pair` | FunctionDef | См. реализацию | [L10](../../../../tests/test_cc_v7_population.py#L10) |
| `test_population_excludes_shared_reference_without_claiming_a_new_scene` | FunctionDef | См. реализацию | [L14](../../../../tests/test_cc_v7_population.py#L14) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_population_excludes_shared_reference_without_claiming_a_new_scene` · L14

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_population_excludes_shared_reference_without_claiming_a_new_scene():
    old=[pair("a","image0","reference0")]
    new=[pair("b","image1","reference0"),pair("c","image2","reference2")]
    rows=select(new,old)
    assert [r["primary"] for r in rows]==[False,True]
    assert all(r["role"]=="external_test_only" for r in rows)
    with pytest.raises(ValueError,match="Historical image overlap"):
        select([pair("b","image0","reference2")],old)
    with pytest.raises(ValueError,match="Duplicate"):
        select(new+new,old)
```

</details>
