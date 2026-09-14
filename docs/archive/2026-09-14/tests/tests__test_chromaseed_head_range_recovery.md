# `tests/test_chromaseed_head_range_recovery.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_head_range_recovery.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `55dd8ca556791233eb24eba6065057fd8dd8e8e6a6a64b1970049f911b5cbd6b`. Строк: **75**.

## Зависимости

```python
import json
import sys
from pathlib import Path
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_transient_windows_denial_retries_same_payload` | FunctionDef | См. реализацию | [L10](../../../../tests/test_chromaseed_head_range_recovery.py#L10) |
| `test_only_exhausted_progress_is_nonfatal_and_critical_writes_still_fail` | FunctionDef | См. реализацию | [L30](../../../../tests/test_chromaseed_head_range_recovery.py#L30) |
| `test_non_permission_failures_are_never_swallowed` | FunctionDef | См. реализацию | [L47](../../../../tests/test_chromaseed_head_range_recovery.py#L47) |
| `test_original_writer_reproduces_sharing_failure_then_recovers` | FunctionDef | См. реализацию | [L59](../../../../tests/test_chromaseed_head_range_recovery.py#L59) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_transient_windows_denial_retries_same_payload` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_transient_windows_denial_retries_same_payload(tmp_path):
    from chromaseed_head_range_recovery import retry_writer

    calls, waits, events = [], [], []
    payload = {'step': 128, 'loss': [1., 2.]}
    path = tmp_path / 'progress.json'

    def writer(p, value):
        calls.append((p, value))
        if len(calls) < 3:
            raise PermissionError('Windows reader holds destination')
        p.write_text(json.dumps(value))

    wrapped = retry_writer(writer, path, delays=(.01, .02), sleep=waits.append, report=events.append)
    wrapped(path, payload)
    assert calls == [(path, payload)]*3 and all(v is payload for _, v in calls)
    assert waits == [.01, .02] and json.loads(path.read_text()) == payload
    assert events[-1]['outcome'] == 'retried' and events[-1]['retries'] == 2
```

</details>

### `test_only_exhausted_progress_is_nonfatal_and_critical_writes_still_fail` · L30

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_only_exhausted_progress_is_nonfatal_and_critical_writes_still_fail(tmp_path):
    from chromaseed_head_range_recovery import retry_writer

    events = []

    def denied(path, value):
        raise PermissionError('busy')

    progress = tmp_path / 'progress.json'
    wrapped = retry_writer(denied, progress, delays=(), report=events.append)
    assert wrapped(progress, {'step': 128}) is None
    assert events[-1]['outcome'] == 'telemetry_skipped'
    with pytest.raises(PermissionError):
        wrapped(tmp_path / 'receipt.json', {'models': 'required'})
    assert events[-1]['outcome'] == 'critical_failed'
```

</details>

### `test_non_permission_failures_are_never_swallowed` · L47

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_non_permission_failures_are_never_swallowed(tmp_path):
    from chromaseed_head_range_recovery import retry_writer

    def invalid(path, value):
        raise ValueError('nonfinite result')

    progress = tmp_path / 'progress.json'
    with pytest.raises(ValueError, match='nonfinite result'):
        retry_writer(invalid, progress)(progress, {})
```

</details>

### `test_original_writer_reproduces_sharing_failure_then_recovers` · L59

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.skipif(sys.platform != 'win32', reason='Real Windows file sharing semantics')
def test_original_writer_reproduces_sharing_failure_then_recovers(tmp_path):
    from chromaseed_head_range_recovery import retry_writer
    from skin_local_search_train import write_json

    path = tmp_path / 'progress.json'
    write_json(path, {'old': 1})
    handle = path.open('r', encoding='utf-8')
    try:
        with pytest.raises(PermissionError):
            write_json(path, {'new': 2})
        events = []
        wrapped = retry_writer(write_json, path, delays=(.01,), sleep=lambda _: handle.close(), report=events.append)
        wrapped(path, {'new': 2})
        assert json.loads(path.read_text()) == {'new': 2}
        assert events[-1]['retries'] == 1
    finally:
        handle.close()
```

</details>
