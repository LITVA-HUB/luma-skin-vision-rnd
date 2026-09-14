# `scripts/chromaseed_widen_early_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_widen_early_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Earlier and lower-rate WIDE training with exact original and paired canaries.

SHA-256 исходника: `a7f500bc5d32500927135e466f731f54ed9d271408239f0ccc4aa3281b0f76d1`. Строк: **440**.

## Зависимости

```python
from __future__ import annotations
import os
import time
import numpy as np
import torch
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import context, save_npz
from chromaseed_neural_prefix_numpy import predict as base_predict
from chromaseed_patch8_numpy import choose, transform_tokens
from chromaseed_refine_train import setup
from chromaseed_widen import SEEDS, capacity, predict
from chromaseed_widen_audit import exact
from chromaseed_widen_early_fit import FIRST_RATES, fit
from chromaseed_widen_run import OUT as WIDE_OUT
from chromaseed_widen_run import ROOT, check_map, load_data
from chromaseed_widen_run import RUN as WIDE
from chromaseed_widen_run import bank_path as wide_bank
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 26](../../../../scripts/chromaseed_widen_early_run.py#L26)

```python
CAPS = ("m31", "m61", "m111", "m832")
```

[Строка 27](../../../../scripts/chromaseed_widen_early_run.py#L27)

```python
TIMES = (0, 32, 128, 512, 2048)
```

[Строка 28](../../../../scripts/chromaseed_widen_early_run.py#L28)

```python
RUN = ROOT / "experiments/runs/chromaseed_widen_early_v1"
```

[Строка 29](../../../../scripts/chromaseed_widen_early_run.py#L29)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_widen_early_v1"
```

[Строка 30](../../../../scripts/chromaseed_widen_early_run.py#L30)

```python
PARENT = "d86795ebc9659fdefe869298fa8a44106555e5ac90a5fa153105d408be3b4876"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bank_path` | FunctionDef | См. реализацию | [L33](../../../../scripts/chromaseed_widen_early_run.py#L33) |
| `freeze` | FunctionDef | См. реализацию | [L44](../../../../scripts/chromaseed_widen_early_run.py#L44) |
| `train_bank` | FunctionDef | См. реализацию | [L95](../../../../scripts/chromaseed_widen_early_run.py#L95) |
| `oof` | FunctionDef | См. реализацию | [L190](../../../../scripts/chromaseed_widen_early_run.py#L190) |
| `select` | FunctionDef | См. реализацию | [L204](../../../../scripts/chromaseed_widen_early_run.py#L204) |
| `selected_sources` | FunctionDef | См. реализацию | [L261](../../../../scripts/chromaseed_widen_early_run.py#L261) |
| `evaluate` | FunctionDef | См. реализацию | [L342](../../../../scripts/chromaseed_widen_early_run.py#L342) |
| `main` | FunctionDef | См. реализацию | [L411](../../../../scripts/chromaseed_widen_early_run.py#L411) |
