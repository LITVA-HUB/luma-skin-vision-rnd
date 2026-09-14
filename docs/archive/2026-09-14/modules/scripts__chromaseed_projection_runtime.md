# `scripts/chromaseed_projection_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_projection_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Actual standalone X response and complete-fit cost, sequential after audit.

SHA-256 исходника: `1dbfb81f197573bef5f59bd0f560c5c9d2a7ead611b6a1bb191782327da051e4`. Строк: **190**.

## Зависимости

```python
from __future__ import annotations
import argparse
import gc
import os
import platform
import time
from pathlib import Path
import numpy as np
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_projection import FAMILIES, fit_single, model_id
from chromaseed_projection_numpy import Predictor
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_projection_runtime.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `query_time` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_projection_runtime.py#L22) |
| `timing_fit` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_projection_runtime.py#L54) |
| `main` | FunctionDef | См. реализацию | [L84](../../../../scripts/chromaseed_projection_runtime.py#L84) |
