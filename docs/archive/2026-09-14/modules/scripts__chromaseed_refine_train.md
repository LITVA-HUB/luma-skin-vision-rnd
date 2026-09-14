# `scripts/chromaseed_refine_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_refine_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Resumable TRAIN-only ChromaSeed-R experiment; separate selection/evaluation.

SHA-256 исходника: `4ac8ad22c7c246be91fdd2eaf2fa2c58ad1a9366c2ce1c9a15a726b395ff5fde`. Строк: **483**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
from chromaseed_refine import (
    FAMILIES,
    BankAdamW,
    BankNet,
    apply_exit_policy,
    fit_preprocessor,
    transform,
)
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    metrics,
    roles,
    sha,
    synchronize,
    weights_for,
    write_json,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_refine_train.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 38](../../../../scripts/chromaseed_refine_train.py#L38)

```python
SEEDS = (17, 29, 43)
```

[Строка 39](../../../../scripts/chromaseed_refine_train.py#L39)

```python
LRS = (.0003, .001, .003)
```

[Строка 40](../../../../scripts/chromaseed_refine_train.py#L40)

```python
CHECKPOINTS = (512, 2048, 8192)
```

[Строка 41](../../../../scripts/chromaseed_refine_train.py#L41)

```python
THRESHOLDS = (0., .25, .5, 1.)
```

[Строка 42](../../../../scripts/chromaseed_refine_train.py#L42)

```python
SLOTS = tuple((seed, lr) for seed in SEEDS for lr in LRS)
```

[Строка 43](../../../../scripts/chromaseed_refine_train.py#L43)

```python
_WARMUP_STREAM = None
```

[Строка 44](../../../../scripts/chromaseed_refine_train.py#L44)

```python
BOUND_SOURCES = (
    "scripts/chromaseed_refine.py", "scripts/chromaseed_refine_train.py",
    "scripts/skin_local_search_train.py", "scripts/skin_local_search_core.py",
    "src/luma_skin_vision/color.py", "docs/research/chromaseed_refine_v1_protocol.md",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `setup` | FunctionDef | См. реализацию | [L51](../../../../scripts/chromaseed_refine_train.py#L51) |
| `read_json` | FunctionDef | См. реализацию | [L61](../../../../scripts/chromaseed_refine_train.py#L61) |
| `atomic_npz` | FunctionDef | См. реализацию | [L65](../../../../scripts/chromaseed_refine_train.py#L65) |
| `source_lock` | FunctionDef | См. реализацию | [L73](../../../../scripts/chromaseed_refine_train.py#L73) |
| `load_data` | FunctionDef | См. реализацию | [L93](../../../../scripts/chromaseed_refine_train.py#L93) |
| `sampling_indices` | FunctionDef | См. реализацию | [L103](../../../../scripts/chromaseed_refine_train.py#L103) |
| `train_bank` | FunctionDef | См. реализацию | [L115](../../../../scripts/chromaseed_refine_train.py#L115) |
| `predict_bank` | FunctionDef | См. реализацию | [L217](../../../../scripts/chromaseed_refine_train.py#L217) |
| `completed_bank` | FunctionDef | См. реализацию | [L233](../../../../scripts/chromaseed_refine_train.py#L233) |
| `progress_callback` | FunctionDef | См. реализацию | [L246](../../../../scripts/chromaseed_refine_train.py#L246) |
| `inner_stage` | FunctionDef | См. реализацию | [L257](../../../../scripts/chromaseed_refine_train.py#L257) |
| `read_oof` | FunctionDef | См. реализацию | [L293](../../../../scripts/chromaseed_refine_train.py#L293) |
| `seed_averaged_policy` | FunctionDef | См. реализацию | [L307](../../../../scripts/chromaseed_refine_train.py#L307) |
| `select_stage` | FunctionDef | См. реализацию | [L319](../../../../scripts/chromaseed_refine_train.py#L319) |
| `final_stage` | FunctionDef | См. реализацию | [L356](../../../../scripts/chromaseed_refine_train.py#L356) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L389](../../../../scripts/chromaseed_refine_train.py#L389) |
| `smoke` | FunctionDef | См. реализацию | [L437](../../../../scripts/chromaseed_refine_train.py#L437) |
| `main` | FunctionDef | См. реализацию | [L454](../../../../scripts/chromaseed_refine_train.py#L454) |
