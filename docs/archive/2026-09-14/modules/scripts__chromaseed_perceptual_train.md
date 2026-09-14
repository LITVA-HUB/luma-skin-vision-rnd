# `scripts/chromaseed_perceptual_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_perceptual_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen person-disjoint perceptual readout study, using original TRAIN only.

SHA-256 исходника: `c9f42dd4b2a8ac947497fd7f2dc77b3fc2742bbd28e650df377390ab3b617519`. Строк: **236**.

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
from chromaseed_fast_kernel import flatten_bank, get_model, model_id
from chromaseed_fast_kernel_train import average_metrics, js, nz, row_hash, valid_bank
from chromaseed_kernel import coordinates, gaussian_kernel, predict_kernel
from chromaseed_kernel_train import atomic_npz
from chromaseed_perceptual import ALPHAS, CHECKPOINTS, FAMILIES, RANK, SEEDS, WIDTHS, fit_bank, key
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

[Строка 28](../../../../scripts/chromaseed_perceptual_train.py#L28)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `candidate_grid` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_perceptual_train.py#L31) |
| `choose_candidate` | FunctionDef | См. реализацию | [L39](../../../../scripts/chromaseed_perceptual_train.py#L39) |
| `lock_sources` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_perceptual_train.py#L43) |
| `control` | FunctionDef | См. реализацию | [L69](../../../../scripts/chromaseed_perceptual_train.py#L69) |
| `predict_bank` | FunctionDef | См. реализацию | [L89](../../../../scripts/chromaseed_perceptual_train.py#L89) |
| `run_bank` | FunctionDef | См. реализацию | [L103](../../../../scripts/chromaseed_perceptual_train.py#L103) |
| `fit_stage` | FunctionDef | См. реализацию | [L128](../../../../scripts/chromaseed_perceptual_train.py#L128) |
| `select_stage` | FunctionDef | См. реализацию | [L144](../../../../scripts/chromaseed_perceptual_train.py#L144) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L175](../../../../scripts/chromaseed_perceptual_train.py#L175) |
| `main` | FunctionDef | См. реализацию | [L206](../../../../scripts/chromaseed_perceptual_train.py#L206) |
