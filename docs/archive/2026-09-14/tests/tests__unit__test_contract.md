# `tests/unit/test_contract.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/unit/test_contract.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `282a712d1ec79ff79d83285d9fe250a868f4736457b7dd314119dad56f1d5042`. Строк: **29**.

## Зависимости

```python
import pytest
from luma_skin_vision.selective import decide
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_policy` | FunctionDef | См. реализацию | [L19](../../../../tests/unit/test_contract.py#L19) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_policy` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('flags,domain,calibrated,upper,expected', [([], False, True, 1, 'UNSUPPORTED'), ([], True, False, 1, 'UNSUPPORTED'), (['blur'], True, True, 1, 'RETAKE'), ([], True, True, 7, 'RETAKE'), ([], True, True, 3, 'ACCEPT'), ([], True, True, float('nan'), 'RETAKE'), ([], True, True, float('inf'), 'RETAKE'), ([], True, True, -1, 'RETAKE')])
def test_policy(flags, domain, calibrated, upper, expected):
    assert (
        decide(
            quality_flags=flags,
            domain_supported=domain,
            calibrated=calibrated,
            upper_error=upper,
            tolerance=5,
        )
        == expected
    )
```

</details>
