# `scripts/chromaseed_gaussian_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gaussian_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Time75 real consumers and72 complete individual TG/FG fits after audit.

SHA-256 исходника: `7156df8bc6bc00a91244d37deeaa175f8279c2282640e71432de5bf758c6b5b8`. Строк: **199**.

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
from chromaseed_feature_groups import fit_single as fg_fit
from chromaseed_feature_groups_numpy import Predictor as FGPredictor
from chromaseed_gaussian import fit_single
from chromaseed_gaussian_numpy import Predictor
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 20](../../../../scripts/chromaseed_gaussian_runtime.py#L20)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `consumer` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_gaussian_runtime.py#L23) |
| `query_time` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_gaussian_runtime.py#L27) |
| `timing_fit` | FunctionDef | См. реализацию | [L58](../../../../scripts/chromaseed_gaussian_runtime.py#L58) |
| `main` | FunctionDef | См. реализацию | [L108](../../../../scripts/chromaseed_gaussian_runtime.py#L108) |
