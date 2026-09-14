# `scripts/chromaseed_weak_ridge_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_weak_ridge_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Expanded-ridge banks with complete exact-P control preservation.

SHA-256 исходника: `9a9e540e461919a9c494a19354a51cef9111c483aee1171a7ab34ae1e823d042`. Строк: **248**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import platform
import time
from pathlib import Path
import numpy as np
import scipy
import torch
from chromaseed_fast_kernel import flatten_bank, get_model
from chromaseed_fast_kernel_train import average_metrics, js, nz, row_hash, valid_bank
from chromaseed_kernel import coordinates, gaussian_kernel, predict_kernel
from chromaseed_kernel_train import atomic_npz
from chromaseed_weak_ridge import ALPHAS, CHECKPOINTS, FAMILIES, RANK, SEEDS, WIDTHS, fit_bank, key
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    metrics,
    roles,
    sha,
    weights_for,
    write_json,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 28](../../../../scripts/chromaseed_weak_ridge_train.py#L28)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `candidate_grid` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_weak_ridge_train.py#L31) |
| `choose_candidate` | FunctionDef | См. реализацию | [L39](../../../../scripts/chromaseed_weak_ridge_train.py#L39) |
| `lock_sources` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_weak_ridge_train.py#L43) |
| `control` | FunctionDef | См. реализацию | [L81](../../../../scripts/chromaseed_weak_ridge_train.py#L81) |
| `predict_bank` | FunctionDef | См. реализацию | [L99](../../../../scripts/chromaseed_weak_ridge_train.py#L99) |
| `run_bank` | FunctionDef | См. реализацию | [L113](../../../../scripts/chromaseed_weak_ridge_train.py#L113) |
| `fit_stage` | FunctionDef | См. реализацию | [L138](../../../../scripts/chromaseed_weak_ridge_train.py#L138) |
| `select_stage` | FunctionDef | См. реализацию | [L154](../../../../scripts/chromaseed_weak_ridge_train.py#L154) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L185](../../../../scripts/chromaseed_weak_ridge_train.py#L185) |
| `main` | FunctionDef | См. реализацию | [L216](../../../../scripts/chromaseed_weak_ridge_train.py#L216) |
