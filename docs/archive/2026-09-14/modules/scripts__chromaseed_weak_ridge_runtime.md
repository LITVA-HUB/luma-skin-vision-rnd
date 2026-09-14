# `scripts/chromaseed_weak_ridge_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_weak_ridge_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Expanded-ridge full-fit and frozen-predictor timing after independent audit.

SHA-256 исходника: `015e19fd16ebb5b1a27cfd78bb80566271b958e840cd4109f13538c6ac8fd9f8`. Строк: **92**.

## Зависимости

```python
from __future__ import annotations
import argparse
import platform
import time
from pathlib import Path
import numpy as np
import torch
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_runtime import time_queries
from chromaseed_weak_ridge import fit_single
from skin_local_search_train import CACHE_HASH, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_weak_ridge_runtime.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `timed_fit` | FunctionDef | См. реализацию | [L19](../../../../scripts/chromaseed_weak_ridge_runtime.py#L19) |
| `runtime` | FunctionDef | См. реализацию | [L36](../../../../scripts/chromaseed_weak_ridge_runtime.py#L36) |
