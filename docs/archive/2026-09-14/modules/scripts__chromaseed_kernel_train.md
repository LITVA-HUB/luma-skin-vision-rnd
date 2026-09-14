# `scripts/chromaseed_kernel_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_kernel_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Original-TRAIN-only kernel study; choices precede all outer evaluation.

SHA-256 исходника: `ab47976c0b444cfa862e6779ecd46f21ddbdc504ac3d38e8c54fafefcd3f88b3`. Строк: **393**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
from chromaseed_kernel import AdaptiveKernel, convex_blend, predict_kernel
from chromaseed_kernel_bank import (
    ALPHAS,
    FAMILIES,
    RANKS,
    SEEDS,
    WIDTHS,
    choose_from_trace,
    evaluate_bank,
    fit_bank,
    flatten_bank,
    get_model,
    make_adaptive,
    model_id,
    seeds_for,
)
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    metrics,
    roles,
    sha,
    weights_for,
    write_json,
)
from skin_local_search_train import predict as legacy_predict
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_kernel_train.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 47](../../../../scripts/chromaseed_kernel_train.py#L47)

```python
TOLERANCES = (0., .1, .25, .5, 1.)
```

[Строка 48](../../../../scripts/chromaseed_kernel_train.py#L48)

```python
RHOS = (0., .25, .5, .75, 1.)
```

[Строка 49](../../../../scripts/chromaseed_kernel_train.py#L49)

```python
ROLE_NAMES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 50](../../../../scripts/chromaseed_kernel_train.py#L50)

```python
BOUND_SOURCES = ("scripts/chromaseed_kernel.py", "scripts/chromaseed_kernel_bank.py", "scripts/chromaseed_kernel_train.py",
                 "scripts/skin_local_search_train.py", "scripts/skin_local_search_core.py", "src/luma_skin_vision/color.py",
                 "docs/research/chromaseed_kernel_v1_protocol.md")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read_json` | FunctionDef | См. реализацию | [L55](../../../../scripts/chromaseed_kernel_train.py#L55) |
| `read_npz` | FunctionDef | См. реализацию | [L59](../../../../scripts/chromaseed_kernel_train.py#L59) |
| `atomic_npz` | FunctionDef | См. реализацию | [L64](../../../../scripts/chromaseed_kernel_train.py#L64) |
| `row_hash` | FunctionDef | См. реализацию | [L72](../../../../scripts/chromaseed_kernel_train.py#L72) |
| `legacy_path` | FunctionDef | См. реализацию | [L76](../../../../scripts/chromaseed_kernel_train.py#L76) |
| `source_lock` | FunctionDef | См. реализацию | [L89](../../../../scripts/chromaseed_kernel_train.py#L89) |
| `load_data` | FunctionDef | См. реализацию | [L119](../../../../scripts/chromaseed_kernel_train.py#L119) |
| `verify_bank` | FunctionDef | См. реализацию | [L126](../../../../scripts/chromaseed_kernel_train.py#L126) |
| `run_bank` | FunctionDef | См. реализацию | [L138](../../../../scripts/chromaseed_kernel_train.py#L138) |
| `fit_stage` | FunctionDef | См. реализацию | [L168](../../../../scripts/chromaseed_kernel_train.py#L168) |
| `load_oof` | FunctionDef | См. реализацию | [L189](../../../../scripts/chromaseed_kernel_train.py#L189) |
| `average_metrics` | FunctionDef | См. реализацию | [L205](../../../../scripts/chromaseed_kernel_train.py#L205) |
| `legacy_oof` | FunctionDef | См. реализацию | [L210](../../../../scripts/chromaseed_kernel_train.py#L210) |
| `select_stage` | FunctionDef | См. реализацию | [L226](../../../../scripts/chromaseed_kernel_train.py#L226) |
| `numerical_bytes` | FunctionDef | См. реализацию | [L282](../../../../scripts/chromaseed_kernel_train.py#L282) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L286](../../../../scripts/chromaseed_kernel_train.py#L286) |
| `main` | FunctionDef | См. реализацию | [L368](../../../../scripts/chromaseed_kernel_train.py#L368) |
