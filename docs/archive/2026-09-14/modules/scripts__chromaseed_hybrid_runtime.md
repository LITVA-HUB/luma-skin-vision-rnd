# `scripts/chromaseed_hybrid_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_hybrid_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Sequential actual H NumPy consumers and complete single-model timing fits.

SHA-256 исходника: `d4db6c36421b96ba96aabbf1308c84956318a09c6cf2b465b89ec02e4dfd9314`. Строк: **184**.

## Зависимости

```python
from __future__ import annotations
import argparse
import gc
import os
import platform
import time
from pathlib import Path
import numpy as np
from chromaseed_affine_audit import exact
from chromaseed_gated_audit import model_from
from chromaseed_hybrid import fit_single, model_id
from chromaseed_hybrid_numpy import Predictor
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 20](../../../../scripts/chromaseed_hybrid_runtime.py#L20)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `query_time` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_hybrid_runtime.py#L23) |
| `timing_fit` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_hybrid_runtime.py#L54) |
| `main` | FunctionDef | См. реализацию | [L78](../../../../scripts/chromaseed_hybrid_runtime.py#L78) |
