# `scripts/chromaseed_kernel_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_kernel_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Actual batch-one execution and unamortized fit timing for frozen K models.

SHA-256 исходника: `d4db58024ba45bcb44ba0769c902e6ed33124e6da16497fd8bae7f72adde23a8`. Строк: **224**.

## Зависимости

```python
from __future__ import annotations
import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from chromaseed_kernel import (
    AdaptiveKernel,
    coordinates,
    exact_coefficients,
    fit_normalizer,
    gaussian_kernel,
    median_width,
    nystrom_coefficients,
    predict_kernel,
    projection_coefficients,
    select_landmarks,
)
from chromaseed_kernel_bank import ALPHAS, WIDTHS
from skin_local_search_train import CACHE_HASH, roles, sha, weights_for, write_json
from skin_local_search_train import predict as legacy_predict
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 30](../../../../scripts/chromaseed_kernel_runtime.py#L30)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L33](../../../../scripts/chromaseed_kernel_runtime.py#L33) |
| `nz` | FunctionDef | См. реализацию | [L37](../../../../scripts/chromaseed_kernel_runtime.py#L37) |
| `check_lock` | FunctionDef | См. реализацию | [L42](../../../../scripts/chromaseed_kernel_runtime.py#L42) |
| `filename` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_kernel_runtime.py#L49) |
| `predictor` | FunctionDef | См. реализацию | [L56](../../../../scripts/chromaseed_kernel_runtime.py#L56) |
| `time_queries` | FunctionDef | См. реализацию | [L74](../../../../scripts/chromaseed_kernel_runtime.py#L74) |
| `standalone_fit` | FunctionDef | См. реализацию | [L102](../../../../scripts/chromaseed_kernel_runtime.py#L102) |
| `measure` | FunctionDef | См. реализацию | [L125](../../../../scripts/chromaseed_kernel_runtime.py#L125) |
| `main` | FunctionDef | См. реализацию | [L210](../../../../scripts/chromaseed_kernel_runtime.py#L210) |
