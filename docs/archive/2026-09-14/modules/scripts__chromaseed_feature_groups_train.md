# `scripts/chromaseed_feature_groups_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_feature_groups_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Registered TRAIN-only feature groups: inner selection precedes final fits.

SHA-256 исходника: `9a8d0992e9b78abdee67d29274fdf7d43801500fa55c9d62817edecd4e9f5fc9`. Строк: **474**.

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
from chromaseed_affine import model_id as a_id
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_feature_groups import (
    ALL_GROUPS,
    ALPHAS,
    FAMILIES,
    GROUPS,
    POLICIES,
    SEEDS,
    choose_alpha,
    choose_policy,
    fit_bank,
    gate_score,
    model_id,
    predict,
)
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_projection import model_id as x_id
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

[Строка 46](../../../../scripts/chromaseed_feature_groups_train.py#L46)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 47](../../../../scripts/chromaseed_feature_groups_train.py#L47)

```python
PARENT_A = ROOT / "experiments/runs/chromaseed_affine_v1"
```

[Строка 48](../../../../scripts/chromaseed_feature_groups_train.py#L48)

```python
PARENT_X = ROOT / "experiments/runs/chromaseed_projection_v1"
```

[Строка 49](../../../../scripts/chromaseed_feature_groups_train.py#L49)

```python
PARENT_C = ROOT / "experiments/runs/chromaseed_crossfit_v1"
```

[Строка 50](../../../../scripts/chromaseed_feature_groups_train.py#L50)

```python
C_VERIFICATION = "80c4129565931e204d663d4e833d8513e7d70111b47c965b6ffe63f2eb2c18dd"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `exact` | FunctionDef | См. реализацию | [L53](../../../../scripts/chromaseed_feature_groups_train.py#L53) |
| `lock_sources` | FunctionDef | См. реализацию | [L60](../../../../scripts/chromaseed_feature_groups_train.py#L60) |
| `parent_bank` | FunctionDef | См. реализацию | [L125](../../../../scripts/chromaseed_feature_groups_train.py#L125) |
| `add_controls` | FunctionDef | См. реализацию | [L134](../../../../scripts/chromaseed_feature_groups_train.py#L134) |
| `run_bank` | FunctionDef | См. реализацию | [L176](../../../../scripts/chromaseed_feature_groups_train.py#L176) |
| `fit_stage` | FunctionDef | См. реализацию | [L256](../../../../scripts/chromaseed_feature_groups_train.py#L256) |
| `score` | FunctionDef | См. реализацию | [L278](../../../../scripts/chromaseed_feature_groups_train.py#L278) |
| `select_stage` | FunctionDef | См. реализацию | [L286](../../../../scripts/chromaseed_feature_groups_train.py#L286) |
| `evaluation_cases` | FunctionDef | См. реализацию | [L342](../../../../scripts/chromaseed_feature_groups_train.py#L342) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L373](../../../../scripts/chromaseed_feature_groups_train.py#L373) |
| `main` | FunctionDef | См. реализацию | [L442](../../../../scripts/chromaseed_feature_groups_train.py#L442) |
