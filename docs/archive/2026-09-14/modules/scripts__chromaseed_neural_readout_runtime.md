# `scripts/chromaseed_neural_readout_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_readout_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Actual NR consumer profiling and full, uncached model construction repeats.

SHA-256 исходника: `40c6084006a88f3e41db5419a557d5c1cc273269f31ed8b5bd130b6ace8cd464`. Строк: **184**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import platform
import time
from pathlib import Path
import numpy as np
from chromaseed_feature_groups import fit_single as fg_fit
from chromaseed_gaussian_runtime import consumer, query_time
from chromaseed_kernel_audit import js, nz
from chromaseed_neural_readout import fit_single
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/chromaseed_neural_readout_runtime.py#L18)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `timing_fit` | FunctionDef | См. реализацию | [L21](../../../../scripts/chromaseed_neural_readout_runtime.py#L21) |
| `main` | FunctionDef | См. реализацию | [L81](../../../../scripts/chromaseed_neural_readout_runtime.py#L81) |
