# `scripts/chromaseed_neural_blocks_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_blocks_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched full/subset training banks to8192; immutable NP policies unchanged.

SHA-256 исходника: `476632301872efb866c3e40ce88d0ec69ec70d9e40c91c6022526f76443134fa`. Строк: **374**.

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
from chromaseed_local_denoise_fit import fit as full_fit
from chromaseed_local_denoise_train import CACHE, ROOT, load_data, save_npz
from chromaseed_neural_blocks_fit import fit as subset_fit
from chromaseed_neural_blocks_fit import to_prefix
from chromaseed_neural_prefix_numpy import SPECS, capacity, export_prefix, predict
from chromaseed_neural_prefix_run import RUN as NP
from chromaseed_neural_prefix_run import check_map
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_train import setup
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 26](../../../../scripts/chromaseed_neural_blocks_run.py#L26)

```python
RUN = ROOT / "experiments/runs/chromaseed_neural_blocks_v1"
```

[Строка 27](../../../../scripts/chromaseed_neural_blocks_run.py#L27)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_neural_blocks_v1"
```

[Строка 28](../../../../scripts/chromaseed_neural_blocks_run.py#L28)

```python
NP_OUT = ROOT / "docs/benchmarks/chromaseed_neural_prefix_v1"
```

[Строка 30](../../../../scripts/chromaseed_neural_blocks_run.py#L30)

```python
FAMILIES = ("local2", "local4", "blind4")
```

[Строка 31](../../../../scripts/chromaseed_neural_blocks_run.py#L31)

```python
PARENT_HASH = "f9bd98ddad578678278581ce2fde786ced806e13e1db3a15cea1aa0fd33fe024"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `settings` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_neural_blocks_run.py#L34) |
| `freeze` | FunctionDef | См. реализацию | [L57](../../../../scripts/chromaseed_neural_blocks_run.py#L57) |
| `bank_path` | FunctionDef | См. реализацию | [L102](../../../../scripts/chromaseed_neural_blocks_run.py#L102) |
| `train_bank` | FunctionDef | См. реализацию | [L112](../../../../scripts/chromaseed_neural_blocks_run.py#L112) |
| `compare_arrays` | FunctionDef | См. реализацию | [L181](../../../../scripts/chromaseed_neural_blocks_run.py#L181) |
| `outputs` | FunctionDef | См. реализацию | [L196](../../../../scripts/chromaseed_neural_blocks_run.py#L196) |
| `evaluate` | FunctionDef | См. реализацию | [L200](../../../../scripts/chromaseed_neural_blocks_run.py#L200) |
| `main` | FunctionDef | См. реализацию | [L335](../../../../scripts/chromaseed_neural_blocks_run.py#L335) |
