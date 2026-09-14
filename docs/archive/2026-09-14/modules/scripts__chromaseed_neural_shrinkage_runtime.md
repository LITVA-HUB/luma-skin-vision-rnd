# `scripts/chromaseed_neural_shrinkage_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_shrinkage_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Actual NS inference and270 full model reconstructions without cached bases.

SHA-256 исходника: `a39077b8bcce35d9c95794be0909cd1828b15e8df09bc158d6e0a5bd38b8bd1b`. Строк: **127**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import numpy as np
from chromaseed_gaussian_runtime import query_time
from chromaseed_kernel_audit import js, nz
from chromaseed_neural_readout_runtime import timing_fit
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_neural_shrinkage_runtime.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_neural_shrinkage_runtime.py#L17)

```python
RUN = ROOT / "experiments/runs/chromaseed_neural_shrinkage_v1"
```

[Строка 18](../../../../scripts/chromaseed_neural_shrinkage_runtime.py#L18)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_neural_shrinkage_v1"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L21](../../../../scripts/chromaseed_neural_shrinkage_runtime.py#L21) |
