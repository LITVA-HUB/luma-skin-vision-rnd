# `tests/test_cc_v3_data.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v3_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `f3d330bdd6c7a0427500854dbf8edaa449ab28863d6f36d35029720498f95599`. Строк: **37**.

## Зависимости

```python
import sys
from pathlib import Path
import pytest
from cc_v3_data import remainder
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sample` | FunctionDef | См. реализацию | [L10](../../../../tests/test_cc_v3_data.py#L10) |
| `test_remainder_is_exact_disjoint_metadata_population` | FunctionDef | См. реализацию | [L19](../../../../tests/test_cc_v3_data.py#L19) |
| `test_remainder_refuses_ambiguous_or_changed_historical_members` | FunctionDef | См. реализацию | [L28](../../../../tests/test_cc_v3_data.py#L28) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_remainder_is_exact_disjoint_metadata_population` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_remainder_is_exact_disjoint_metadata_population():
    result = remainder([sample(i) for i in range(256)], [sample(i) for i in range(128)])
    assert len(result) == 128
    assert [s["image"]["path"] for s in result] == [
        sample(i)["image"]["path"] for i in range(128, 256)
    ]
```

</details>

### `test_remainder_refuses_ambiguous_or_changed_historical_members` · L28

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'reordered_metadata'])
def test_remainder_refuses_ambiguous_or_changed_historical_members(kind):
    all_rows, old = [sample(i) for i in range(256)], [sample(i) for i in range(128)]
    if kind == "duplicate":
        all_rows[-1] = all_rows[-2]
    elif kind == "missing":
        old[-1] = sample(999)
    else:
        old[-1]["gt"]["path"] = "different.wp"
    with pytest.raises(ValueError):
        remainder(all_rows, old)
```

</details>
