# `tests/unit/test_response_schema.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/unit/test_response_schema.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `f9b2561ffe3f0ba8bf8c9b1fda3a60ca91cb5ba281de2b96e8f82c24b866cf43`. Строк: **20**.

## Зависимости

```python
import pytest
from luma_skin_vision.contracts import AnalysisResponse
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_accept_cannot_be_empty_or_synthetic` | FunctionDef | См. реализацию | [L6](../../../../tests/unit/test_response_schema.py#L6) |
| `test_unsupported_contract_defaults_are_conservative` | FunctionDef | См. реализацию | [L17](../../../../tests/unit/test_response_schema.py#L17) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_accept_cannot_be_empty_or_synthetic` · L6

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_accept_cannot_be_empty_or_synthetic():
    with pytest.raises(ValueError):
        AnalysisResponse(
            schema_version="1.0",
            status="ACCEPT",
            measurement=None,
            evidence_kind="SYNTHETIC",
            model_version="smoke",
        )
```

</details>

### `test_unsupported_contract_defaults_are_conservative` · L17

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_unsupported_contract_defaults_are_conservative():
    response = AnalysisResponse(model_version="smoke", evidence_kind="SYNTHETIC")
    assert response.measurement is None
    assert response.profile_update["may_update"] is False
```

</details>
