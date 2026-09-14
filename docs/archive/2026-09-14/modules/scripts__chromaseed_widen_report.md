# `scripts/chromaseed_widen_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_widen_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Publish local verified quality/size evidence; later invocations only reverify hashes.

SHA-256 исходника: `a14120a36c61c450c27c99c7e400aca3cdc62f6d8ddf6a966c31ce7dbcdcd003`. Строк: **333**.

## Зависимости

```python
from __future__ import annotations
import subprocess
import sys
import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import NP
from chromaseed_refine_audit import error_summary
from chromaseed_widen_run import OUT, ROOT, RUN, bank_path, check_map, load_data
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/chromaseed_widen_report.py#L15)

```python
ROLE_NAMES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 16](../../../../scripts/chromaseed_widen_report.py#L16)

```python
VARIANTS = ("np", "tiny", "m31", "m61", "m111", "m832")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L19](../../../../scripts/chromaseed_widen_report.py#L19) |
