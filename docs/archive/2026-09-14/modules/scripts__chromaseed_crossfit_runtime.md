# `scripts/chromaseed_crossfit_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_crossfit_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Actual unchanged H consumer, including full teacher-pool costs in C fits.

SHA-256 исходника: `724e1a096fce2064ef28b6980dfe453c08dadd94f7289ac9238b47924ae087a6`. Строк: **146**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import platform
import time
from pathlib import Path
import numpy as np
from chromaseed_affine_audit import exact
from chromaseed_crossfit import fit_single as c_fit
from chromaseed_hybrid import fit_single as h_fit
from chromaseed_hybrid_runtime import query_time
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_crossfit_runtime.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `timing_fit` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_crossfit_runtime.py#L22) |
| `main` | FunctionDef | См. реализацию | [L51](../../../../scripts/chromaseed_crossfit_runtime.py#L51) |
