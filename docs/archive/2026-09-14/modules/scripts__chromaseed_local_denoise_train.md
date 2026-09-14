# `scripts/chromaseed_local_denoise_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_local_denoise_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen, resumable ND nested selection and target-free evaluation.

SHA-256 исходника: `b5c0efc84fd1307b749b96a8071eb0dd915075be2ddc0f69ffeea57935396764`. Строк: **492**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from chromaseed_fast_kernel_train import row_hash
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gaussian_train import infer
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_local_denoise import SPECS, Bank
from chromaseed_local_denoise_fit import fit
from chromaseed_local_denoise_numpy import predict
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_train import setup
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

[Строка 34](../../../../scripts/chromaseed_local_denoise_train.py#L34)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 35](../../../../scripts/chromaseed_local_denoise_train.py#L35)

```python
RUN = ROOT / "experiments/runs/chromaseed_local_denoise_v1"
```

[Строка 36](../../../../scripts/chromaseed_local_denoise_train.py#L36)

```python
NS = ROOT / "experiments/runs/chromaseed_neural_shrinkage_v1"
```

[Строка 37](../../../../scripts/chromaseed_local_denoise_train.py#L37)

```python
CACHE = ROOT.parents[1] / "luma-skin-vision-rnd/data/processed/skin_mskcc_pixels_v1/train.npz"
```

[Строка 39](../../../../scripts/chromaseed_local_denoise_train.py#L39)

```python
SLOTS = tuple((seed, lr) for seed in SEEDS for lr in LRS)
```

[Строка 40](../../../../scripts/chromaseed_local_denoise_train.py#L40)

```python
PARENT_HASH = "e725b4a6e86df7a51dd9825089ca39c218100bc3e31f765a585946353fb9cd6f"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `load_data` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_local_denoise_train.py#L43) |
| `references` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_local_denoise_train.py#L49) |
| `freeze` | FunctionDef | См. реализацию | [L58](../../../../scripts/chromaseed_local_denoise_train.py#L58) |
| `save_npz` | FunctionDef | См. реализацию | [L133](../../../../scripts/chromaseed_local_denoise_train.py#L133) |
| `ready` | FunctionDef | См. реализацию | [L143](../../../../scripts/chromaseed_local_denoise_train.py#L143) |
| `export_drift` | FunctionDef | См. реализацию | [L154](../../../../scripts/chromaseed_local_denoise_train.py#L154) |
| `train_one` | FunctionDef | См. реализацию | [L171](../../../../scripts/chromaseed_local_denoise_train.py#L171) |
| `inner` | FunctionDef | См. реализацию | [L247](../../../../scripts/chromaseed_local_denoise_train.py#L247) |
| `oof` | FunctionDef | См. реализацию | [L268](../../../../scripts/chromaseed_local_denoise_train.py#L268) |
| `select` | FunctionDef | См. реализацию | [L277](../../../../scripts/chromaseed_local_denoise_train.py#L277) |
| `final` | FunctionDef | См. реализацию | [L318](../../../../scripts/chromaseed_local_denoise_train.py#L318) |
| `evaluate` | FunctionDef | См. реализацию | [L338](../../../../scripts/chromaseed_local_denoise_train.py#L338) |
| `main` | FunctionDef | См. реализацию | [L441](../../../../scripts/chromaseed_local_denoise_train.py#L441) |
