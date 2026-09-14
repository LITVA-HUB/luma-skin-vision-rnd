# `scripts/chromaseed_gated_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gated_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen banks and person-held-out selection for compact conditional residuals.

SHA-256 исходника: `40c695c223e96099a241bd42b3ca2c812bfdb62e98accad48708fa6f067c8930`. Строк: **313**.

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
from chromaseed_fast_kernel import get_model
from chromaseed_fast_kernel_train import average_metrics, js, nz, row_hash, valid_bank
from chromaseed_gated import (
    FAMILIES,
    SEEDS,
    candidates,
    fit_bank,
    flatten,
    model_id,
    predict,
    unpack,
)
from chromaseed_kernel_train import atomic_npz
from skin_local_search_train import CACHE_HASH, folds_for, metrics, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 28](../../../../scripts/chromaseed_gated_train.py#L28)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `lock_sources` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_gated_train.py#L31) |
| `check_controls` | FunctionDef | См. реализацию | [L79](../../../../scripts/chromaseed_gated_train.py#L79) |
| `run_bank` | FunctionDef | См. реализацию | [L96](../../../../scripts/chromaseed_gated_train.py#L96) |
| `fit_stage` | FunctionDef | См. реализацию | [L155](../../../../scripts/chromaseed_gated_train.py#L155) |
| `choose` | FunctionDef | См. реализацию | [L179](../../../../scripts/chromaseed_gated_train.py#L179) |
| `select_stage` | FunctionDef | См. реализацию | [L183](../../../../scripts/chromaseed_gated_train.py#L183) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L216](../../../../scripts/chromaseed_gated_train.py#L216) |
| `main` | FunctionDef | См. реализацию | [L280](../../../../scripts/chromaseed_gated_train.py#L280) |
