# `scripts/chromaseed_hybrid_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_hybrid_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

H frozen shared-center training, person-held-out selection and fixed stress evaluation.

SHA-256 исходника: `7b5e0b2df2800e921c35322eaddc662954eba25189d42cb31ea997954dd95d80`. Строк: **436**.

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
from chromaseed_affine import model_id as a_id
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_gate_stability import affine_features, gate_score, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_hybrid import (
    ALPHAS,
    KINDS,
    LOSSES,
    POWERS,
    RHOS,
    SEEDS,
    choose,
    fit_bank,
    model_id,
    predict,
    settings,
)
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_projection import model_id as x_id
from chromaseed_projection_train import parent_bank, score
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

[Строка 45](../../../../scripts/chromaseed_hybrid_train.py#L45)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 46](../../../../scripts/chromaseed_hybrid_train.py#L46)

```python
PARENT_A = ROOT / "experiments/runs/chromaseed_affine_v1"
```

[Строка 47](../../../../scripts/chromaseed_hybrid_train.py#L47)

```python
PARENT_X = ROOT / "experiments/runs/chromaseed_projection_v1"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `lock_sources` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_hybrid_train.py#L50) |
| `controls` | FunctionDef | См. реализацию | [L118](../../../../scripts/chromaseed_hybrid_train.py#L118) |
| `run_bank` | FunctionDef | См. реализацию | [L147](../../../../scripts/chromaseed_hybrid_train.py#L147) |
| `fit_stage` | FunctionDef | См. реализацию | [L213](../../../../scripts/chromaseed_hybrid_train.py#L213) |
| `select_stage` | FunctionDef | См. реализацию | [L235](../../../../scripts/chromaseed_hybrid_train.py#L235) |
| `cases` | FunctionDef | См. реализацию | [L295](../../../../scripts/chromaseed_hybrid_train.py#L295) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L342](../../../../scripts/chromaseed_hybrid_train.py#L342) |
| `main` | FunctionDef | См. реализацию | [L406](../../../../scripts/chromaseed_hybrid_train.py#L406) |
