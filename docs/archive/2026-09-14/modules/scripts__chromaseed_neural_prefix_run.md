# `scripts/chromaseed_neural_prefix_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_prefix_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Immutable conditional NP inner selection followed by all-prefix evaluation.

SHA-256 исходника: `9708f55600ae265771df6dd0bd34a14475f0cf5c0a253813b34303dd6b500425`. Строк: **261**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import time
import numpy as np
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import unpack
from chromaseed_gaussian_train import infer
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_train import CACHE, ROOT, SEEDS, load_data, oof, save_npz
from chromaseed_local_denoise_train import RUN as ND
from chromaseed_neural_prefix_numpy import (
    SPECS,
    TOLERANCE,
    capacity,
    choose,
    export_prefix,
    predict,
)
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 26](../../../../scripts/chromaseed_neural_prefix_run.py#L26)

```python
RUN = ROOT / "experiments/runs/chromaseed_neural_prefix_v1"
```

[Строка 27](../../../../scripts/chromaseed_neural_prefix_run.py#L27)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_neural_prefix_v1"
```

[Строка 28](../../../../scripts/chromaseed_neural_prefix_run.py#L28)

```python
ND_OUT = ROOT / "docs/benchmarks/chromaseed_local_denoise_v1"
```

[Строка 29](../../../../scripts/chromaseed_neural_prefix_run.py#L29)

```python
PARENT_HASH = "36442da6dabe927b4056ef5532fc7a6dd3956c14eb8ed35736a3f161ac3e2376"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `check_map` | FunctionDef | См. реализацию | [L32](../../../../scripts/chromaseed_neural_prefix_run.py#L32) |
| `freeze` | FunctionDef | См. реализацию | [L37](../../../../scripts/chromaseed_neural_prefix_run.py#L37) |
| `select` | FunctionDef | См. реализацию | [L88](../../../../scripts/chromaseed_neural_prefix_run.py#L88) |
| `evaluate` | FunctionDef | См. реализацию | [L138](../../../../scripts/chromaseed_neural_prefix_run.py#L138) |
| `main` | FunctionDef | См. реализацию | [L227](../../../../scripts/chromaseed_neural_prefix_run.py#L227) |
