# `scripts/chromaseed_gated_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gated_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Sequential canonical/standalone inference and complete-fit timing after G audit.

SHA-256 исходника: `c32008cc8e38e22b581d89cd3d6f4b1ca9a05306cb1fbee8b4b5a42a93074045`. Строк: **197**.

## Зависимости

```python
from __future__ import annotations
import argparse
import gc
import platform
import time
from pathlib import Path
import numpy as np
from chromaseed_gated import fit_single, model_id, predict, unpack
from chromaseed_gated_numpy import Predictor
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_gated_runtime.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `query_time` | FunctionDef | См. реализацию | [L20](../../../../scripts/chromaseed_gated_runtime.py#L20) |
| `timing_fit` | FunctionDef | См. реализацию | [L52](../../../../scripts/chromaseed_gated_runtime.py#L52) |
| `main` | FunctionDef | См. реализацию | [L83](../../../../scripts/chromaseed_gated_runtime.py#L83) |
