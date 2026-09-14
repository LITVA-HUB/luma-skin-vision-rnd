# `scripts/chromaseed_gaussian_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gaussian_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Freeze TG, train nested traces, choose settings, then fit and evaluate final models.

SHA-256 исходника: `b8cb59ae93b250d2873eee21e36d9d50d8dbe92a233c5af016ff7eedd3e1bf51`. Строк: **459**.

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
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_feature_groups import predict as fg_predict
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gaussian import (
    CHECKPOINTS,
    GROUPS,
    METHODS,
    PARAMETERS,
    SEEDS,
    choose,
    model_id,
    predict,
    train_block,
)
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

[Строка 42](../../../../scripts/chromaseed_gaussian_train.py#L42)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 43](../../../../scripts/chromaseed_gaussian_train.py#L43)

```python
FG = ROOT / "experiments/runs/chromaseed_feature_groups_v1"
```

[Строка 44](../../../../scripts/chromaseed_gaussian_train.py#L44)

```python
FG_VERIFICATION = "459e6554bfba7ce22c63f7377556648632ba07406d20e1d50a42351d8bbf8908"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `lock_sources` | FunctionDef | См. реализацию | [L47](../../../../scripts/chromaseed_gaussian_train.py#L47) |
| `controls` | FunctionDef | См. реализацию | [L118](../../../../scripts/chromaseed_gaussian_train.py#L118) |
| `infer` | FunctionDef | См. реализацию | [L155](../../../../scripts/chromaseed_gaussian_train.py#L155) |
| `run_bank` | FunctionDef | См. реализацию | [L159](../../../../scripts/chromaseed_gaussian_train.py#L159) |
| `fit_stage` | FunctionDef | См. реализацию | [L260](../../../../scripts/chromaseed_gaussian_train.py#L260) |
| `score` | FunctionDef | См. реализацию | [L283](../../../../scripts/chromaseed_gaussian_train.py#L283) |
| `select_stage` | FunctionDef | См. реализацию | [L291](../../../../scripts/chromaseed_gaussian_train.py#L291) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L351](../../../../scripts/chromaseed_gaussian_train.py#L351) |
| `main` | FunctionDef | См. реализацию | [L427](../../../../scripts/chromaseed_gaussian_train.py#L427) |
