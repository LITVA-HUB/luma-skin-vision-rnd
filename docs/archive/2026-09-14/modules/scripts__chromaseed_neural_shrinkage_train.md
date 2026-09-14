# `scripts/chromaseed_neural_shrinkage_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_shrinkage_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

One registered stronger-alpha study on exact NR representations.

SHA-256 исходника: `16f4c450301d55358d23d9f7c4ea78b4c6ad2e678a0839fe90139d709564a849`. Строк: **406**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import numpy as np
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gaussian_audit import direct
from chromaseed_gaussian_train import infer, score
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_neural_readout import (
    BASES,
    FAMILIES,
    GROUPS,
    SEEDS,
    choose,
    choose_policy,
    fit_head,
    name_for,
)
from chromaseed_neural_readout_reference import refit
from chromaseed_perceptual_audit import balanced
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

[Строка 40](../../../../scripts/chromaseed_neural_shrinkage_train.py#L40)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 41](../../../../scripts/chromaseed_neural_shrinkage_train.py#L41)

```python
NR = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
```

[Строка 42](../../../../scripts/chromaseed_neural_shrinkage_train.py#L42)

```python
NG = ROOT / "experiments/runs/chromaseed_neural_geometry_v1"
```

[Строка 43](../../../../scripts/chromaseed_neural_shrinkage_train.py#L43)

```python
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
```

[Строка 44](../../../../scripts/chromaseed_neural_shrinkage_train.py#L44)

```python
NR_HASH = "3e4079df5d688136ef59be6b599633b0bc7c44d9d21f410eaeefc2aef0699c65"
```

[Строка 45](../../../../scripts/chromaseed_neural_shrinkage_train.py#L45)

```python
NG_HASH = "57b1d6b00b51e9f761ad34b2062f28a1186cd403f0ba40a52e24b0d105014e25"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `freeze` | FunctionDef | См. реализацию | [L48](../../../../scripts/chromaseed_neural_shrinkage_train.py#L48) |
| `bank` | FunctionDef | См. реализацию | [L101](../../../../scripts/chromaseed_neural_shrinkage_train.py#L101) |
| `fit_stage` | FunctionDef | См. реализацию | [L218](../../../../scripts/chromaseed_neural_shrinkage_train.py#L218) |
| `select` | FunctionDef | См. реализацию | [L240](../../../../scripts/chromaseed_neural_shrinkage_train.py#L240) |
| `evaluate` | FunctionDef | См. реализацию | [L311](../../../../scripts/chromaseed_neural_shrinkage_train.py#L311) |
| `main` | FunctionDef | См. реализацию | [L369](../../../../scripts/chromaseed_neural_shrinkage_train.py#L369) |
