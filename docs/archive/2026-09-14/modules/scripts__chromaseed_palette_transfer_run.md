# `scripts/chromaseed_palette_transfer_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_transfer_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

P3 preregistration and matched native run, gated on completed and sealed HR.

SHA-256 исходника: `269edbfab95c7fb25e9988aabde939521dfb6e766713e9efe8ca7322b85047c3`. Строк: **629**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from chromaseed_architecture_scale import BATCH, HORIZON, RATES, SEEDS, SLOTS, capacity
from chromaseed_gated import flatten, unpack
from chromaseed_head_range import predict_torch
from chromaseed_head_range_audit import bank_path as hr_bank
from chromaseed_head_range_verification import (
    CONTRACT as HR_CONTRACT,
)
from chromaseed_head_range_verification import (
    OUT as HR_OUT,
)
from chromaseed_head_range_verification import (
    ROLES,
    ROOT,
    check_hashes,
    digest,
    read,
    require_quiet_host,
    require_terminal,
    write_once,
)
from chromaseed_head_range_verification import (
    RUN as HR_RUN,
)
from chromaseed_head_range_verification import (
    SOURCE as HR_SOURCE,
)
from chromaseed_kernel_audit import nz
from chromaseed_long_training_run import context
from chromaseed_palette_transfer import ARMS, VARIANTS, validate_encoders
from chromaseed_palette_transfer_fit import fit
from chromaseed_patch8_numpy import choose
from chromaseed_refine_train import setup
from chromaseed_widen_run import load_data
from skin_local_search_train import metrics, roles, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 48](../../../../scripts/chromaseed_palette_transfer_run.py#L48)

```python
RUN = Path("D:/Luma-RnD/chromaseed_palette_transfer_v1")
```

[Строка 49](../../../../scripts/chromaseed_palette_transfer_run.py#L49)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_palette_transfer_v1"
```

[Строка 50](../../../../scripts/chromaseed_palette_transfer_run.py#L50)

```python
P2 = Path("D:/Luma-RnD/chromaseed_palette_pretrain_v1")
```

[Строка 51](../../../../scripts/chromaseed_palette_transfer_run.py#L51)

```python
TIMES = (128, 512, 2048)
```

[Строка 52](../../../../scripts/chromaseed_palette_transfer_run.py#L52)

```python
FILES = (
    "scripts/chromaseed_palette_transfer.py",
    "scripts/chromaseed_palette_transfer_fit.py",
    "scripts/chromaseed_palette_transfer_run.py",
    "scripts/chromaseed_palette_transfer_preflight.py",
    "tests/test_chromaseed_palette_transfer.py",
    "docs/research/chromaseed_palette_transfer_v1_protocol.md",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `palette_bindings` | FunctionDef | См. реализацию | [L62](../../../../scripts/chromaseed_palette_transfer_run.py#L62) |
| `encoders_for` | FunctionDef | См. реализацию | [L90](../../../../scripts/chromaseed_palette_transfer_run.py#L90) |
| `resolve_heads` | FunctionDef | См. реализацию | [L103](../../../../scripts/chromaseed_palette_transfer_run.py#L103) |
| `register` | FunctionDef | См. реализацию | [L120](../../../../scripts/chromaseed_palette_transfer_run.py#L120) |
| `verify_registration` | FunctionDef | См. реализацию | [L149](../../../../scripts/chromaseed_palette_transfer_run.py#L149) |
| `require_hr_sealed` | FunctionDef | См. реализацию | [L155](../../../../scripts/chromaseed_palette_transfer_run.py#L155) |
| `freeze` | FunctionDef | См. реализацию | [L172](../../../../scripts/chromaseed_palette_transfer_run.py#L172) |
| `bank_path` | FunctionDef | См. реализацию | [L212](../../../../scripts/chromaseed_palette_transfer_run.py#L212) |
| `mutable_json` | FunctionDef | См. реализацию | [L222](../../../../scripts/chromaseed_palette_transfer_run.py#L222) |
| `save_arrays` | FunctionDef | См. реализацию | [L235](../../../../scripts/chromaseed_palette_transfer_run.py#L235) |
| `train_bank` | FunctionDef | См. реализацию | [L241](../../../../scripts/chromaseed_palette_transfer_run.py#L241) |
| `policies` | FunctionDef | См. реализацию | [L342](../../../../scripts/chromaseed_palette_transfer_run.py#L342) |
| `select` | FunctionDef | См. реализацию | [L356](../../../../scripts/chromaseed_palette_transfer_run.py#L356) |
| `evaluate` | FunctionDef | См. реализацию | [L432](../../../../scripts/chromaseed_palette_transfer_run.py#L432) |
| `main` | FunctionDef | См. реализацию | [L537](../../../../scripts/chromaseed_palette_transfer_run.py#L537) |
