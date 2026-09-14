# `scripts/chromaseed_projection_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_projection_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen X representation banks, inner quality/compact policies and fixed-dose finals.

SHA-256 исходника: `3418b2496e55aef38b50ea467d2f78e62d94a9f39f79360086a730711fd31b45`. Строк: **505**.

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
from chromaseed_gate_stability import affine_features, gate_score, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gated import model_id as g_id
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_projection import (
    ALPHAS,
    FAMILIES,
    POLICIES,
    REPRESENTATIONS,
    SEEDS,
    choose,
    fit_bank,
    model_id,
    predict,
)
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

[Строка 43](../../../../scripts/chromaseed_projection_train.py#L43)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 44](../../../../scripts/chromaseed_projection_train.py#L44)

```python
PARENT_A = ROOT / "experiments/runs/chromaseed_affine_v1"
```

[Строка 45](../../../../scripts/chromaseed_projection_train.py#L45)

```python
PARENT_G = ROOT / "experiments/runs/chromaseed_gated_v1"
```

[Строка 46](../../../../scripts/chromaseed_projection_train.py#L46)

```python
CONTROLS = (
    "g_norm_soft",
    "g_perceptual_soft",
    "a_norm_joint_guarded",
    "a_perceptual_joint_guarded",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `lock_sources` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_projection_train.py#L54) |
| `parent_bank` | FunctionDef | См. реализацию | [L140](../../../../scripts/chromaseed_projection_train.py#L140) |
| `controls` | FunctionDef | См. реализацию | [L149](../../../../scripts/chromaseed_projection_train.py#L149) |
| `run_bank` | FunctionDef | См. реализацию | [L198](../../../../scripts/chromaseed_projection_train.py#L198) |
| `fit_stage` | FunctionDef | См. реализацию | [L277](../../../../scripts/chromaseed_projection_train.py#L277) |
| `score` | FunctionDef | См. реализацию | [L299](../../../../scripts/chromaseed_projection_train.py#L299) |
| `select_stage` | FunctionDef | См. реализацию | [L307](../../../../scripts/chromaseed_projection_train.py#L307) |
| `evaluation_cases` | FunctionDef | См. реализацию | [L377](../../../../scripts/chromaseed_projection_train.py#L377) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L407](../../../../scripts/chromaseed_projection_train.py#L407) |
| `main` | FunctionDef | См. реализацию | [L475](../../../../scripts/chromaseed_projection_train.py#L475) |
