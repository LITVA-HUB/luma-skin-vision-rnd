# `apps/training-dashboard/test_monitor.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../apps/training-dashboard/test_monitor.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

The monitor must distinguish persisted facts from live estimates.

SHA-256 исходника: `ec3ce4ab1b3d034e601a138edea5129c6a97f159c42222449ba7a3826d918d9a`. Строк: **135**.

## Зависимости

```python
import csv
import io
import json
import os
import sys
from pathlib import Path
from monitor import ROLES, VARIANTS, Monitor
from server import export_csv
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `write` | FunctionDef | См. реализацию | [L15](../../../../apps/training-dashboard/test_monitor.py#L15) |
| `fixture` | FunctionDef | См. реализацию | [L21](../../../../apps/training-dashboard/test_monitor.py#L21) |
| `receipt` | FunctionDef | См. реализацию | [L34](../../../../apps/training-dashboard/test_monitor.py#L34) |
| `test_full_schedule_and_estimates_do_not_become_completed_facts` | FunctionDef | См. реализацию | [L58](../../../../apps/training-dashboard/test_monitor.py#L58) |
| `test_stale_running_job_is_not_reported_as_live` | FunctionDef | См. реализацию | [L73](../../../../apps/training-dashboard/test_monitor.py#L73) |
| `test_progress_never_claims_full_completion_without_a_receipt` | FunctionDef | См. реализацию | [L83](../../../../apps/training-dashboard/test_monitor.py#L83) |
| `test_final_checkpoint_changes_are_used_in_duration_estimates` | FunctionDef | См. реализацию | [L92](../../../../apps/training-dashboard/test_monitor.py#L92) |
| `test_malformed_metadata_returns_degraded_state_not_fake_progress` | FunctionDef | См. реализацию | [L104](../../../../apps/training-dashboard/test_monitor.py#L104) |
| `test_missing_gpu_reading_is_unavailable_not_zero` | FunctionDef | См. реализацию | [L115](../../../../apps/training-dashboard/test_monitor.py#L115) |
| `test_csv_uses_current_filters_and_preserves_russian_text` | FunctionDef | См. реализацию | [L123](../../../../apps/training-dashboard/test_monitor.py#L123) |

## Все тестовые определения (7)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_full_schedule_and_estimates_do_not_become_completed_facts` · L58

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_full_schedule_and_estimates_do_not_become_completed_facts(tmp_path):
    monitor, run = fixture(tmp_path)
    receipt(run)
    data = monitor.snapshot()
    assert len(data["packages"]) == 84
    assert data["progress"]["inner_total"] == 63
    assert data["progress"]["final_total"] == 21
    assert data["progress"]["completed"] == 1
    assert data["current"]["id"] == "inner/mixed/patch_small/fold1"
    assert data["current"]["estimated"]
    assert 0 < data["current"]["step"] < 2048
    assert data["gpu"]["utilization"] == 88
    assert data["traces"][0]["points"][0]["losses"] == [1, 2, 3, 4, 5, 6]
```

</details>

### `test_stale_running_job_is_not_reported_as_live` · L73

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_stale_running_job_is_not_reported_as_live(tmp_path):
    monitor, run = fixture(tmp_path, alive=False)
    receipt(run)
    data = monitor.snapshot()
    assert data["state"] == "interrupted"
    assert data["current"] is None
    assert data["eta"]["seconds"] is None
    assert not any(p["status"] == "running" for p in data["packages"])
```

</details>

### `test_progress_never_claims_full_completion_without_a_receipt` · L83

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_progress_never_claims_full_completion_without_a_receipt(tmp_path):
    monitor, run = fixture(tmp_path, now=100_000)
    receipt(run)
    data = monitor.snapshot()
    assert data["current"]["step"] < data["current"]["target_steps"]
    assert data["current"]["overdue"]
    assert data["progress"]["completed"] == 1
```

</details>

### `test_final_checkpoint_changes_are_used_in_duration_estimates` · L92

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_final_checkpoint_changes_are_used_in_duration_estimates(tmp_path):
    monitor, run = fixture(tmp_path)
    receipt(run)
    selection = {"roles": {r: {"policies": {v: {"step": 128} for v in VARIANTS}} for r in ROLES}}
    write(run / "selections.json", selection, 1200)
    data = monitor.snapshot()
    finals = [p for p in data["packages"] if p["stage"] == "final"]
    assert len(finals) == 21 and all(p["target_steps"] == 128 for p in finals)
    assert all(not p["steps_unknown"] for p in finals)
    assert next(p for p in finals if p["variant"] == "patch_small")["expected_seconds"] < 100
```

</details>

### `test_malformed_metadata_returns_degraded_state_not_fake_progress` · L104

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_malformed_metadata_returns_degraded_state_not_fake_progress(tmp_path):
    monitor, run = fixture(tmp_path)
    p = run / "inner/mixed/patch_small/fold0/receipt.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"broken":', encoding="utf-8")
    data = monitor.snapshot()
    assert data["warnings"]
    assert data["state"] == "degraded"
    assert data["current"] is None
```

</details>

### `test_missing_gpu_reading_is_unavailable_not_zero` · L115

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_missing_gpu_reading_is_unavailable_not_zero(tmp_path):
    monitor, _ = fixture(tmp_path)
    monitor.gpu_probe = lambda: {"available": False, "error": "unavailable"}
    data = monitor.snapshot()
    assert not data["gpu"]["available"]
    assert data["gpu"].get("utilization") is None
```

</details>

### `test_csv_uses_current_filters_and_preserves_russian_text` · L123

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_csv_uses_current_filters_and_preserves_russian_text(tmp_path):
    monitor, run = fixture(tmp_path)
    receipt(run)
    payload = export_csv(
        monitor.snapshot()["packages"],
        {"status": "completed", "role": "mixed", "query": "PATCH_SMALL"},
    )
    assert payload.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig")), delimiter=";"))
    assert len(rows) == 2
    assert rows[0][0] == "Архитектура"
    assert rows[1][0] == "patch_small"
    assert rows[1][-2:] == ["Завершён", "Нет"]
```

</details>
