# `apps/training-dashboard/test_facial_monitor.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../apps/training-dashboard/test_facial_monitor.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `6acd6971bf8159be081506c165e7c8554b45e14b5ba7a47756f7e30ae70ffdf8`. Строк: **25**.

## Зависимости

```python
import json
import sys
from pathlib import Path
from facial_monitor import snapshot
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_facial_monitor_reads_actual_data_and_marks_dead_job` | FunctionDef | См. реализацию | [L9](../../../../apps/training-dashboard/test_facial_monitor.py#L9) |
| `test_facial_monitor_missing_data_is_not_mocked` | FunctionDef | См. реализацию | [L24](../../../../apps/training-dashboard/test_facial_monitor.py#L24) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_facial_monitor_reads_actual_data_and_marks_dead_job` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_facial_monitor_reads_actual_data_and_marks_dead_job(tmp_path):
    p=tmp_path/'lapa/prepared_192/profile.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(splits=dict(train=dict(rows=15914),val=dict(rows=1692)))))
    run=tmp_path/'facial_skin_v1'
    run.mkdir()
    (run/'progress.json').write_text(json.dumps(dict(status='training',epoch=2)))
    (run/'job.json').write_text(json.dumps(dict(pid=123)))
    row=snapshot(tmp_path,probe=lambda pid:False)
    assert row['train_images']==15914 and row['state']=='interrupted'
    row=snapshot(tmp_path,probe=lambda pid:True)
    assert row['state']=='training'
    assert snapshot(tmp_path,now=10**12,probe=lambda pid:True)['state']=='stale'
```

</details>

### `test_facial_monitor_missing_data_is_not_mocked` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_facial_monitor_missing_data_is_not_mocked(tmp_path):
    assert snapshot(tmp_path) is None
```

</details>
