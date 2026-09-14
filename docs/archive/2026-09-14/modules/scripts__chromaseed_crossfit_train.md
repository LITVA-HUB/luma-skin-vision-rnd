# `scripts/chromaseed_crossfit_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_crossfit_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen inherited H settings, whole-person teachers, matched C heads and evaluation.

SHA-256 исходника: `f362feacb03b8ea441b060268ca0d81debc14097f42a2f5fb76201a4d6df6a07`. Строк: **334**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from chromaseed_affine_audit import exact
from chromaseed_crossfit import ARMS, LOSSES, SEEDS, fit_bank
from chromaseed_gate_stability import affine_features, gate_score, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_hybrid import predict
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 23](../../../../scripts/chromaseed_crossfit_train.py#L23)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 24](../../../../scripts/chromaseed_crossfit_train.py#L24)

```python
PARENT = ROOT / "experiments/runs/chromaseed_hybrid_v1"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `freeze` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_crossfit_train.py#L27) |
| `train` | FunctionDef | См. реализацию | [L116](../../../../scripts/chromaseed_crossfit_train.py#L116) |
| `evaluate` | FunctionDef | См. реализацию | [L235](../../../../scripts/chromaseed_crossfit_train.py#L235) |
| `main` | FunctionDef | См. реализацию | [L308](../../../../scripts/chromaseed_crossfit_train.py#L308) |
