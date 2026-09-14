# `scripts/chromaseed_fast_kernel_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_fast_kernel_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Verified KF inference, independent-fit timing and synthetic allocation scale.

SHA-256 исходника: `a41f5cfdce50c689f8db6a0f1836446944a0db33de930bff15fcd2796bdb0659`. Строк: **140**.

## Зависимости

```python
from __future__ import annotations
import argparse
import gc
import json
import platform
import time
import tracemalloc
from pathlib import Path
import numpy as np
import torch
from chromaseed_fast_kernel import fit_one
from chromaseed_kernel import predict_kernel
from chromaseed_kernel_runtime import time_queries
from skin_local_search_train import CACHE_HASH, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_fast_kernel_runtime.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_fast_kernel_runtime.py#L22) |
| `nz` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_fast_kernel_runtime.py#L26) |
| `locks` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_fast_kernel_runtime.py#L31) |
| `allocation_peak` | FunctionDef | См. реализацию | [L41](../../../../scripts/chromaseed_fast_kernel_runtime.py#L41) |
| `run_measurement` | FunctionDef | См. реализацию | [L51](../../../../scripts/chromaseed_fast_kernel_runtime.py#L51) |
| `main` | FunctionDef | См. реализацию | [L130](../../../../scripts/chromaseed_fast_kernel_runtime.py#L130) |
