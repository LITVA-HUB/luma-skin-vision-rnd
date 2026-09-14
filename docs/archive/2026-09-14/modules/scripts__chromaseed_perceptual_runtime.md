# `scripts/chromaseed_perceptual_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_perceptual_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Standalone timing after audit; no new quality selection or outer scores.

SHA-256 исходника: `f8883afe95c777c846358af6ce5d3ce647aa12210c3b1e4b23847cf480f16fe5`. Строк: **90**.

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
from chromaseed_perceptual import fit_single
from skin_local_search_train import CACHE_HASH, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_perceptual_runtime.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `timed_fit` | FunctionDef | См. реализацию | [L19](../../../../scripts/chromaseed_perceptual_runtime.py#L19) |
| `runtime` | FunctionDef | См. реализацию | [L36](../../../../scripts/chromaseed_perceptual_runtime.py#L36) |
