# `scripts/chromaseed_architecture_scale_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_architecture_scale_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seal the measured AS screen after independent audit and complete training replay.

SHA-256 исходника: `ce693afd88d242c64833ae95767d9b1b3df0c0856dd28292152e74eb4a8b539e`. Строк: **310**.

## Зависимости

```python
from __future__ import annotations
import subprocess
import sys
import numpy as np
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, VARIANTS, WE_OUT, check_map, load_data
from chromaseed_kernel_audit import js, nz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_architecture_scale_report.py#L14)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L17](../../../../scripts/chromaseed_architecture_scale_report.py#L17) |
