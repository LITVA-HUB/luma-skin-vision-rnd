# `tests/test_cc_phone_resume.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_phone_resume.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Retry policy must respect server throttling and preserve other failures.

SHA-256 исходника: `94ad1d946d55b0387a3055052b518cd39d6b06ad4cba9de950f8f4451a13f689`. Строк: **40**.

## Зависимости

```python
import importlib.util
from pathlib import Path
from urllib.error import HTTPError
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `module` | FunctionDef | См. реализацию | [L9](../../../../tests/test_cc_phone_resume.py#L9) |
| `test_rate_limit_is_retried_after_server_requested_delay` | FunctionDef | См. реализацию | [L18](../../../../tests/test_cc_phone_resume.py#L18) |
| `test_missing_file_is_not_retried_or_hidden` | FunctionDef | См. реализацию | [L32](../../../../tests/test_cc_phone_resume.py#L32) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_rate_limit_is_retried_after_server_requested_delay` · L18

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_rate_limit_is_retried_after_server_requested_delay():
    mod = module()
    calls, waits = [], []
    def response(request, **kwargs):
        calls.append((request, kwargs))
        if len(calls) == 1:
            raise HTTPError('https://example.test', 429, 'slow down', {'Retry-After': '120'}, None)
        return b'verified-response'
    opener = mod.polite_opener(response, waits.append)
    assert opener('same-range', timeout=60) == b'verified-response'
    assert calls[0] == calls[1]
    assert sum(waits) >= 120
```

</details>

### `test_missing_file_is_not_retried_or_hidden` · L32

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_missing_file_is_not_retried_or_hidden():
    mod = module()
    calls = []
    def response(*args, **kwargs):
        calls.append(1)
        raise HTTPError('https://example.test', 404, 'missing', {}, None)
    with pytest.raises(HTTPError) as exc:
        mod.polite_opener(response, lambda seconds: None)('range')
    assert exc.value.code == 404 and len(calls) == 1
```

</details>
