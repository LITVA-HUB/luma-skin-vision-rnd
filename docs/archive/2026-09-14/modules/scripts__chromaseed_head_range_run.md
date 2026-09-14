# `scripts/chromaseed_head_range_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_head_range_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched HR sweep; independent new output heads, frozen AS unit controls.

SHA-256 исходника: `39e1251e2fac68a698ae198e1f6bb06475ef02be5b86edd6fab069c7902f5da8`. Строк: **274**.

## Зависимости

```python
import argparse
import os
import shutil
import time
import numpy as np
import torch
from chromaseed_architecture_scale import BATCH, HORIZON, RATES, SEEDS, SLOTS, VARIANTS, capacity
from chromaseed_architecture_scale_run import OUT as AS_OUT
from chromaseed_architecture_scale_run import RUN as AS_RUN
from chromaseed_architecture_scale_run import bank_path as as_bank_path
from chromaseed_architecture_scale_run import save
from chromaseed_gated import flatten, unpack
from chromaseed_head_range import predict_torch
from chromaseed_head_range_fit import fit
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import ROOT, context
from chromaseed_patch8_numpy import choose
from chromaseed_refine_train import setup
from chromaseed_widen_run import check_map, load_data
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 24](../../../../scripts/chromaseed_head_range_run.py#L24)

```python
RUN = ROOT / 'experiments/runs/chromaseed_head_range_v1'
```

[Строка 25](../../../../scripts/chromaseed_head_range_run.py#L25)

```python
OUT = ROOT / 'docs/benchmarks/chromaseed_head_range_v1'
```

[Строка 26](../../../../scripts/chromaseed_head_range_run.py#L26)

```python
TIMES = (128, 512, 2048)
```

[Строка 27](../../../../scripts/chromaseed_head_range_run.py#L27)

```python
MODES = ('unit', 'wide', 'linear')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `pair_key` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_head_range_run.py#L30) |
| `bank_path` | FunctionDef | См. реализацию | [L36](../../../../scripts/chromaseed_head_range_run.py#L36) |
| `freeze` | FunctionDef | См. реализацию | [L42](../../../../scripts/chromaseed_head_range_run.py#L42) |
| `verified_bank` | FunctionDef | См. реализацию | [L79](../../../../scripts/chromaseed_head_range_run.py#L79) |
| `train_bank` | FunctionDef | См. реализацию | [L86](../../../../scripts/chromaseed_head_range_run.py#L86) |
| `oof` | FunctionDef | См. реализацию | [L137](../../../../scripts/chromaseed_head_range_run.py#L137) |
| `policies` | FunctionDef | См. реализацию | [L151](../../../../scripts/chromaseed_head_range_run.py#L151) |
| `select` | FunctionDef | См. реализацию | [L157](../../../../scripts/chromaseed_head_range_run.py#L157) |
| `evaluate` | FunctionDef | См. реализацию | [L192](../../../../scripts/chromaseed_head_range_run.py#L192) |
| `main` | FunctionDef | См. реализацию | [L232](../../../../scripts/chromaseed_head_range_run.py#L232) |
