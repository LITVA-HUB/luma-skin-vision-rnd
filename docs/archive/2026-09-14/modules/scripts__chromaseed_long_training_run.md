# `scripts/chromaseed_long_training_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_long_training_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Nested LT continuation: nine inner banks, frozen choices, three long final banks.

SHA-256 исходника: `fd798176ee7c3e6cfe6f25613f663593f1907aecc1346756579d5a221331382f`. Строк: **406**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import time
import numpy as np
import torch
from chromaseed_fast_kernel_train import row_hash
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_train import CACHE, ROOT, load_data, save_npz
from chromaseed_local_denoise_train import RUN as ND
from chromaseed_long_training_fit import HORIZON, MODES, RATES, SEEDS, SLOTS, fit
from chromaseed_neural_prefix_numpy import export_prefix, predict
from chromaseed_neural_prefix_run import RUN as NP
from chromaseed_neural_prefix_run import check_map
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

[Строка 32](../../../../scripts/chromaseed_long_training_run.py#L32)

```python
RUN = ROOT / "experiments/runs/chromaseed_long_training_v1"
```

[Строка 33](../../../../scripts/chromaseed_long_training_run.py#L33)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_long_training_v1"
```

[Строка 34](../../../../scripts/chromaseed_long_training_run.py#L34)

```python
NB = ROOT / "experiments/runs/chromaseed_neural_blocks_v1"
```

[Строка 35](../../../../scripts/chromaseed_long_training_run.py#L35)

```python
NB_OUT = ROOT / "docs/benchmarks/chromaseed_neural_blocks_v1"
```

[Строка 36](../../../../scripts/chromaseed_long_training_run.py#L36)

```python
CHECKPOINTS = (0, 512, 2048, 8192, 32768, 131072)
```

[Строка 37](../../../../scripts/chromaseed_long_training_run.py#L37)

```python
PARENT_HASH = "671fb2cd20d5e914bbddf3f5f9abafd05a73ea6ecee57c244e769a1cd47f8f2c"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `context` | FunctionDef | См. реализацию | [L40](../../../../scripts/chromaseed_long_training_run.py#L40) |
| `freeze` | FunctionDef | См. реализацию | [L70](../../../../scripts/chromaseed_long_training_run.py#L70) |
| `bank_path` | FunctionDef | См. реализацию | [L116](../../../../scripts/chromaseed_long_training_run.py#L116) |
| `train_one` | FunctionDef | См. реализацию | [L125](../../../../scripts/chromaseed_long_training_run.py#L125) |
| `slot_index` | FunctionDef | См. реализацию | [L205](../../../../scripts/chromaseed_long_training_run.py#L205) |
| `choose` | FunctionDef | См. реализацию | [L209](../../../../scripts/chromaseed_long_training_run.py#L209) |
| `select` | FunctionDef | См. реализацию | [L222](../../../../scripts/chromaseed_long_training_run.py#L222) |
| `evaluate` | FunctionDef | См. реализацию | [L270](../../../../scripts/chromaseed_long_training_run.py#L270) |
| `main` | FunctionDef | См. реализацию | [L357](../../../../scripts/chromaseed_long_training_run.py#L357) |
