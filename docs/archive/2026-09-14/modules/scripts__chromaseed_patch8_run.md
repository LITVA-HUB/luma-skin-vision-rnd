# `scripts/chromaseed_patch8_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_patch8_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Nested P8 local-branch experiment with fixed bank shape and source lineage.

SHA-256 исходника: `fbc9be91a81f0a0369b5c6bb0842ad2224b206715846e44436d0e1fec0eb55f8`. Строк: **364**.

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
from chromaseed_long_training_run import CACHE, ROOT, context, save_npz
from chromaseed_neural_prefix_numpy import predict as base_predict
from chromaseed_neural_prefix_run import check_map
from chromaseed_patch8_fit import ARMS, HORIZON, RATES, SLOTS, Bank, fit, token_normalizers
from chromaseed_patch8_numpy import EXTRA, choose, predict, transform_tokens
from chromaseed_refine_train import setup
from chromaseed_weight_average_run import OUT as WA_OUT
from chromaseed_weight_average_run import RUN as WA
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 24](../../../../scripts/chromaseed_patch8_run.py#L24)

```python
RUN = ROOT / "experiments/runs/chromaseed_patch8_v1"
```

[Строка 25](../../../../scripts/chromaseed_patch8_run.py#L25)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_patch8_v1"
```

[Строка 26](../../../../scripts/chromaseed_patch8_run.py#L26)

```python
CHECKPOINTS = (0, 512, 2048, 8192, 32768)
```

[Строка 27](../../../../scripts/chromaseed_patch8_run.py#L27)

```python
PARENT = "f82f7f7d31b5639abbfef57ce96c79709022205b5d583291cf932cd823b1bdc0"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `load_data` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_patch8_run.py#L30) |
| `bank_path` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_patch8_run.py#L43) |
| `slot_index` | FunctionDef | См. реализацию | [L52](../../../../scripts/chromaseed_patch8_run.py#L52) |
| `freeze` | FunctionDef | См. реализацию | [L56](../../../../scripts/chromaseed_patch8_run.py#L56) |
| `train_one` | FunctionDef | См. реализацию | [L107](../../../../scripts/chromaseed_patch8_run.py#L107) |
| `select` | FunctionDef | См. реализацию | [L171](../../../../scripts/chromaseed_patch8_run.py#L171) |
| `evaluate` | FunctionDef | См. реализацию | [L219](../../../../scripts/chromaseed_patch8_run.py#L219) |
| `main` | FunctionDef | См. реализацию | [L316](../../../../scripts/chromaseed_patch8_run.py#L316) |
