# `tests/test_head_range_monitor.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_head_range_monitor.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `d5bf2da8090282a7c010fe028f6cba870ced23fad635efa1d9a12d2e3353ad2f`. Строк: **35**.

## Зависимости

```python
import json
import sys
from pathlib import Path
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `put` | FunctionDef | См. реализацию | [L8](../../../../tests/test_head_range_monitor.py#L8) |
| `test_dead_worker_cannot_show_live_speed_or_eta` | FunctionDef | См. реализацию | [L14](../../../../tests/test_head_range_monitor.py#L14) |
| `test_forecast_counts_each_registered_architecture_and_respects_failure` | FunctionDef | См. реализацию | [L24](../../../../tests/test_head_range_monitor.py#L24) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_dead_worker_cannot_show_live_speed_or_eta` · L14

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_dead_worker_cannot_show_live_speed_or_eta(tmp_path):
    from head_range_monitor import snapshot
    put(tmp_path,'source_lock.json',{})
    put(tmp_path,'job.json',dict(status='running',pid=123))
    put(tmp_path,'progress.json',dict(status='training',step=128,seconds=10))
    s=snapshot(tmp_path,probe=lambda pid:False)
    assert s['state']=='interrupted' and s['steps_per_second'] is None
    assert s['estimated_remaining_seconds'] is None
```

</details>

### `test_forecast_counts_each_registered_architecture_and_respects_failure` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_forecast_counts_each_registered_architecture_and_respects_failure(tmp_path):
    from head_range_monitor import snapshot
    from monitor import VARIANTS
    put(tmp_path,'source_lock.json',{})
    put(tmp_path,'job.json',dict(status='running',pid=123))
    put(tmp_path,'progress.json',dict(status='training',stage='inner',role='mixed',variant='patch_small',head_mode='wide',fold=0,step=128,seconds=10))
    put(tmp_path,'preflight.json',dict(records=[dict(variant=v,mode=m,info=dict(full_bank_seconds=64)) for v in VARIANTS for m in ['wide','linear']]))
    s=snapshot(tmp_path,probe=lambda pid:True)
    assert s['estimated_remaining_seconds']==168*2048-10
    assert s['steps_per_second']==12.8 and s['completed_banks']==0
    put(tmp_path,'job.json',dict(status='failed',pid=123,error='Recorded failure'))
    assert snapshot(tmp_path,probe=lambda pid:True)['state']=='failed'
```

</details>
