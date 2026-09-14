# `scripts/chromaseed_architecture_scale_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_architecture_scale_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

TRAIN-only architectural screen with selection preceding all held-role access.

SHA-256 исходника: `15f2ad5926f7ec2f5120d5a420def5d30b303dc7c8f4e93a8b4c05ce3c577753`. Строк: **398**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import shutil
import time
import numpy as np
import torch
from chromaseed_architecture_scale import (
    BATCH,
    HORIZON,
    RATES,
    SEEDS,
    SLOTS,
    VARIANTS,
    capacity,
    fit,
    predict_torch,
    specs,
)
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import ROOT, context
from chromaseed_patch8_numpy import choose
from chromaseed_refine_train import setup
from chromaseed_widen_early_run import OUT as WE_OUT
from chromaseed_widen_early_run import RUN as WE
from chromaseed_widen_run import check_map, load_data
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 34](../../../../scripts/chromaseed_architecture_scale_run.py#L34)

```python
RUN = ROOT / "experiments/runs/chromaseed_architecture_scale_v1"
```

[Строка 35](../../../../scripts/chromaseed_architecture_scale_run.py#L35)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_architecture_scale_v1"
```

[Строка 36](../../../../scripts/chromaseed_architecture_scale_run.py#L36)

```python
TIMES = (128, 512, 2048)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `save` | FunctionDef | См. реализацию | [L39](../../../../scripts/chromaseed_architecture_scale_run.py#L39) |
| `bank_path` | FunctionDef | См. реализацию | [L53](../../../../scripts/chromaseed_architecture_scale_run.py#L53) |
| `freeze` | FunctionDef | См. реализацию | [L63](../../../../scripts/chromaseed_architecture_scale_run.py#L63) |
| `train_bank` | FunctionDef | См. реализацию | [L127](../../../../scripts/chromaseed_architecture_scale_run.py#L127) |
| `oof` | FunctionDef | См. реализацию | [L195](../../../../scripts/chromaseed_architecture_scale_run.py#L195) |
| `select` | FunctionDef | См. реализацию | [L203](../../../../scripts/chromaseed_architecture_scale_run.py#L203) |
| `evaluate` | FunctionDef | См. реализацию | [L248](../../../../scripts/chromaseed_architecture_scale_run.py#L248) |
| `preflight` | FunctionDef | См. реализацию | [L325](../../../../scripts/chromaseed_architecture_scale_run.py#L325) |
| `main` | FunctionDef | См. реализацию | [L356](../../../../scripts/chromaseed_architecture_scale_run.py#L356) |
