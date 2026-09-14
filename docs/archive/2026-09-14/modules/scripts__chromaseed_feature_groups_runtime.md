# `scripts/chromaseed_feature_groups_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_feature_groups_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Time real FG batch-one consumers and 384 complete matching fits, sequentially.

SHA-256 исходника: `d51d8794dd537dab7bab8538c1b5aeef6894327a514cbca61e8b4fd2650e76e2`. Строк: **168**.

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
from chromaseed_feature_groups import fit_single
from chromaseed_feature_groups_numpy import Predictor
from chromaseed_kernel_audit import js, nz
from chromaseed_projection import fit_single as x_fit_single
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 20](../../../../scripts/chromaseed_feature_groups_runtime.py#L20)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `query_time` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_feature_groups_runtime.py#L23) |
| `timing_fit` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_feature_groups_runtime.py#L54) |
| `main` | FunctionDef | См. реализацию | [L80](../../../../scripts/chromaseed_feature_groups_runtime.py#L80) |
