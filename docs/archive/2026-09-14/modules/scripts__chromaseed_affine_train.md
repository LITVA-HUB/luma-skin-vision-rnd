# `scripts/chromaseed_affine_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_affine_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen person-disjoint A banks, dual inner policies and all-dose final scoring.

SHA-256 исходника: `0866e1dbd4cbcc54e54579ff2a4b255bb0799916e5c5169b6e0ed2c3a4c31de9`. Строк: **453**.

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
from chromaseed_affine import (
    ALPHAS,
    ETAS,
    FAMILIES,
    G_CONTROLS,
    POLICIES,
    SEEDS,
    choose,
    fit_bank,
    model_id,
    predict,
)
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_gate_stability import affine_features, gate_score, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gated import model_id as g_model_id
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    metrics,
    roles,
    sha,
    weights_for,
    write_json,
)
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 43](../../../../scripts/chromaseed_affine_train.py#L43)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `selection_grid` | FunctionDef | См. реализацию | [L46](../../../../scripts/chromaseed_affine_train.py#L46) |
| `lock_sources` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_affine_train.py#L50) |
| `references` | FunctionDef | См. реализацию | [L121](../../../../scripts/chromaseed_affine_train.py#L121) |
| `run_bank` | FunctionDef | См. реализацию | [L146](../../../../scripts/chromaseed_affine_train.py#L146) |
| `fit_stage` | FunctionDef | См. реализацию | [L213](../../../../scripts/chromaseed_affine_train.py#L213) |
| `score_predictions` | FunctionDef | См. реализацию | [L237](../../../../scripts/chromaseed_affine_train.py#L237) |
| `select_stage` | FunctionDef | См. реализацию | [L247](../../../../scripts/chromaseed_affine_train.py#L247) |
| `evaluation_cases` | FunctionDef | См. реализацию | [L311](../../../../scripts/chromaseed_affine_train.py#L311) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L341](../../../../scripts/chromaseed_affine_train.py#L341) |
| `main` | FunctionDef | См. реализацию | [L419](../../../../scripts/chromaseed_affine_train.py#L419) |
