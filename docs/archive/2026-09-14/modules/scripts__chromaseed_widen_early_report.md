# `scripts/chromaseed_widen_early_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_widen_early_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

WE local result report and immutable read-only verification after sealing.

SHA-256 исходника: `e02317cf108babf36589d947f82ad2820c363a7b051148e101d8b549a3aea336`. Строк: **319**.

## Зависимости

```python
from __future__ import annotations
import subprocess
import sys
import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_refine_audit import error_summary
from chromaseed_widen_early_run import (
    CAPS,
    OUT,
    ROOT,
    RUN,
    WIDE,
    WIDE_OUT,
    check_map,
    load_data,
)
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 23](../../../../scripts/chromaseed_widen_early_report.py#L23)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_widen_early_report.py#L26) |
