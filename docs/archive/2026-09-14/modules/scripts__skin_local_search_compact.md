# `scripts/skin_local_search_compact.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_search_compact.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Secondary frozen-prefix compression study over completed local-search models.

SHA-256 исходника: `684adbfc4c2036dc5a0a8f9dc2f39f1405a2cce3e26fcf5e670bd9ed186c6b23`. Строк: **516**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
from skin_local_search_core import rbf_features, ridge_solve
from skin_local_search_train import (
    CACHE_HASH,
    GRIDS,
    SEEDS,
    load_model,
    metrics,
    sha,
    synchronize,
    weights_for,
    write_json,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/skin_local_search_compact.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 32](../../../../scripts/skin_local_search_compact.py#L32)

```python
FAMILIES = ("random_rbf", "guided_rbf")
```

[Строка 33](../../../../scripts/skin_local_search_compact.py#L33)

```python
PREFIX_COUNTS = (8, 16, 32)
```

[Строка 34](../../../../scripts/skin_local_search_compact.py#L34)

```python
REFERENCE_COUNT = 64
```

[Строка 35](../../../../scripts/skin_local_search_compact.py#L35)

```python
ALL_COUNTS = (*PREFIX_COUNTS, REFERENCE_COUNT)
```

[Строка 36](../../../../scripts/skin_local_search_compact.py#L36)

```python
PROTOCOL = ROOT / "docs/research/skin_local_search_compact_protocol.md"
```

[Строка 37](../../../../scripts/skin_local_search_compact.py#L37)

```python
DEFAULT_SOURCE_RUN = ROOT / "experiments/runs/skin_local_search_v1"
```

[Строка 38](../../../../scripts/skin_local_search_compact.py#L38)

```python
DEFAULT_RUN = ROOT / "experiments/runs/skin_local_search_compact_v1"
```

[Строка 39](../../../../scripts/skin_local_search_compact.py#L39)

```python
FIT_SCRIPT_SHA256 = "f24dfe35370b7c871fcced3540c7b366167b6c85613c34cdac842042fbad3c86"
```

[Строка 40](../../../../scripts/skin_local_search_compact.py#L40)

```python
FIT_PROTOCOL_SHA256 = "950ca0e35380d6484924b8f860d16710d023967e54f8b9d0b1f789f737620e9c"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `_method` | FunctionDef | См. реализацию | [L43](../../../../scripts/skin_local_search_compact.py#L43) |
| `_validate_source_model` | FunctionDef | См. реализацию | [L50](../../../../scripts/skin_local_search_compact.py#L50) |
| `prefix_predict` | FunctionDef | Independent NumPy inference for a compact RBF-prefix payload. | [L75](../../../../scripts/skin_local_search_compact.py#L75) |
| `refit_prefix` | FunctionDef | Refit only the ridge head over the first K atoms of a frozen 64-atom model. | [L108](../../../../scripts/skin_local_search_compact.py#L108) |
| `_verified_model` | FunctionDef | См. реализацию | [L172](../../../../scripts/skin_local_search_compact.py#L172) |
| `_relative` | FunctionDef | См. реализацию | [L182](../../../../scripts/skin_local_search_compact.py#L182) |
| `_load_roles` | FunctionDef | См. реализацию | [L186](../../../../scripts/skin_local_search_compact.py#L186) |
| `_fixed_config` | FunctionDef | См. реализацию | [L192](../../../../scripts/skin_local_search_compact.py#L192) |
| `_lock_sources` | FunctionDef | См. реализацию | [L200](../../../../scripts/skin_local_search_compact.py#L200) |
| `_save_prefix` | FunctionDef | См. реализацию | [L273](../../../../scripts/skin_local_search_compact.py#L273) |
| `fit_phase` | FunctionDef | См. реализацию | [L281](../../../../scripts/skin_local_search_compact.py#L281) |
| `evaluation_phase` | FunctionDef | См. реализацию | [L425](../../../../scripts/skin_local_search_compact.py#L425) |
| `write_evaluation_receipts` | FunctionDef | Persist evaluation output and status without mutating the frozen model manifest. | [L467](../../../../scripts/skin_local_search_compact.py#L467) |
| `main` | FunctionDef | См. реализацию | [L486](../../../../scripts/skin_local_search_compact.py#L486) |
