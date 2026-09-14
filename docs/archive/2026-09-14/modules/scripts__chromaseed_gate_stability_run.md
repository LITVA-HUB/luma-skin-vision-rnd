# `scripts/chromaseed_gate_stability_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gate_stability_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Run frozen G sensitivity and boundary diagnostics on original TRAIN only.

SHA-256 исходника: `26aa1637e0064d1e5d99ae1f97db46097d332a848f2cae0cda4f537b54d66f0f`. Строк: **172**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import numpy as np
from chromaseed_gate_stability import (
    DOSES,
    affine_features,
    boundaries,
    gate_score,
    grid,
    summaries,
)
from chromaseed_gated import predict
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 23](../../../../scripts/chromaseed_gate_stability_run.py#L23)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bind` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_gate_stability_run.py#L26) |
| `main` | FunctionDef | См. реализацию | [L79](../../../../scripts/chromaseed_gate_stability_run.py#L79) |
