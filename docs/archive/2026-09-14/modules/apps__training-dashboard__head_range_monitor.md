# `apps/training-dashboard/head_range_monitor.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../apps/training-dashboard/head_range_monitor.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only HR status and work estimates; no ML dependencies or writers.

SHA-256 исходника: `a0de4b77516f84d3bcdef122e1923d513f05d413c0c52414389a45ae3980b82b`. Строк: **69**.

## Зависимости

```python
import json
import statistics
import time
from pathlib import Path
from monitor import LABELS, ROLES, VARIANTS, process_alive
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `snapshot` | FunctionDef | См. реализацию | [L10](../../../../apps/training-dashboard/head_range_monitor.py#L10) |
