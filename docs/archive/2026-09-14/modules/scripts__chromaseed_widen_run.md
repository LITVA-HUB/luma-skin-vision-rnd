# `scripts/chromaseed_widen_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_widen_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched capacity experiment; inner selection is frozen before held-role evaluation.

SHA-256 исходника: `b4ff2d3acc230ffb6721a0d1718061f372a50c82128827a746a8f3a3b15b6159`. Строк: **389**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import time
import numpy as np
import torch
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import ROOT, context, save_npz
from chromaseed_neural_prefix_numpy import predict as base_predict
from chromaseed_neural_prefix_run import check_map
from chromaseed_patch8_fit import token_normalizers
from chromaseed_patch8_numpy import choose, transform_tokens
from chromaseed_patch8_run import RUN as P8
from chromaseed_patch8_run import load_data
from chromaseed_refine_train import setup
from chromaseed_widen import HORIZON, RATES, SEEDS, SLOTS, SPECS, Bank, capacity, fit, predict
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 25](../../../../scripts/chromaseed_widen_run.py#L25)

```python
RUN = ROOT / "experiments/runs/chromaseed_widen_v1"
```

[Строка 26](../../../../scripts/chromaseed_widen_run.py#L26)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_widen_v1"
```

[Строка 27](../../../../scripts/chromaseed_widen_run.py#L27)

```python
CHECKPOINTS = (0, 512, 2048, 8192)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bank_path` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_widen_run.py#L30) |
| `freeze` | FunctionDef | См. реализацию | [L40](../../../../scripts/chromaseed_widen_run.py#L40) |
| `train_one` | FunctionDef | См. реализацию | [L90](../../../../scripts/chromaseed_widen_run.py#L90) |
| `select` | FunctionDef | См. реализацию | [L161](../../../../scripts/chromaseed_widen_run.py#L161) |
| `evaluate` | FunctionDef | См. реализацию | [L231](../../../../scripts/chromaseed_widen_run.py#L231) |
| `main` | FunctionDef | См. реализацию | [L327](../../../../scripts/chromaseed_widen_run.py#L327) |
