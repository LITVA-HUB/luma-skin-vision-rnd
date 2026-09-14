# `scripts/chromaseed_fast_kernel_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_fast_kernel_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Person-disjoint KF banks, frozen policy selection and isolated evaluation.

SHA-256 исходника: `4f7388b0d85eb049dbc6b54d2a6eecca83b90e98ab1f8234ee1eb4a3c8818adc`. Строк: **299**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from chromaseed_fast_kernel import (
    ALPHAS,
    ARMS,
    PAIR_BUDGETS,
    RANKS,
    SEEDS,
    WIDTHS,
    evaluate_bank,
    fit_bank,
    flatten_bank,
    get_model,
    model_id,
)
from chromaseed_kernel import predict_kernel
from chromaseed_kernel_bank import get_model as old_get_model
from chromaseed_kernel_bank import model_id as old_model_id
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
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 41](../../../../scripts/chromaseed_fast_kernel_train.py#L41)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 42](../../../../scripts/chromaseed_fast_kernel_train.py#L42)

```python
PROTOCOL = ROOT / "docs/research/chromaseed_fast_kernel_v1_protocol.md"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L45](../../../../scripts/chromaseed_fast_kernel_train.py#L45) |
| `nz` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_fast_kernel_train.py#L49) |
| `row_hash` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_fast_kernel_train.py#L54) |
| `choose_fast` | FunctionDef | См. реализацию | [L58](../../../../scripts/chromaseed_fast_kernel_train.py#L58) |
| `choose_size` | FunctionDef | См. реализацию | [L67](../../../../scripts/chromaseed_fast_kernel_train.py#L67) |
| `lock_sources` | FunctionDef | См. реализацию | [L73](../../../../scripts/chromaseed_fast_kernel_train.py#L73) |
| `valid_bank` | FunctionDef | См. реализацию | [L100](../../../../scripts/chromaseed_fast_kernel_train.py#L100) |
| `old_control` | FunctionDef | См. реализацию | [L113](../../../../scripts/chromaseed_fast_kernel_train.py#L113) |
| `run_bank` | FunctionDef | См. реализацию | [L136](../../../../scripts/chromaseed_fast_kernel_train.py#L136) |
| `fit_stage` | FunctionDef | См. реализацию | [L169](../../../../scripts/chromaseed_fast_kernel_train.py#L169) |
| `average_metrics` | FunctionDef | См. реализацию | [L185](../../../../scripts/chromaseed_fast_kernel_train.py#L185) |
| `select_stage` | FunctionDef | См. реализацию | [L190](../../../../scripts/chromaseed_fast_kernel_train.py#L190) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L232](../../../../scripts/chromaseed_fast_kernel_train.py#L232) |
| `main` | FunctionDef | См. реализацию | [L268](../../../../scripts/chromaseed_fast_kernel_train.py#L268) |
