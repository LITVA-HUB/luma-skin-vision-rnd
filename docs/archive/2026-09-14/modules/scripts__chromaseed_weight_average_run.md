# `scripts/chromaseed_weight_average_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_weight_average_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

WA inner-only selection, then final means and matched endpoint controls.

SHA-256 исходника: `15b21071ba5d9fc822b3407ea6807412d8efc3d2cb26df31847f8b935d6b2a95`. Строк: **263**.

## Зависимости

```python
from __future__ import annotations
import os
import time
import numpy as np
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import OUT as LT_OUT
from chromaseed_long_training_run import (
    ROOT,
    bank_path,
    check_map,
    context,
    load_data,
    save_npz,
    slot_index,
)
from chromaseed_long_training_run import RUN as LT
from chromaseed_neural_prefix_numpy import predict
from chromaseed_weight_average import METHODS, STEPS, average, choose, final_union, recipes
from skin_local_search_train import metrics, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 27](../../../../scripts/chromaseed_weight_average_run.py#L27)

```python
RUN = ROOT / "experiments/runs/chromaseed_weight_average_v1"
```

[Строка 28](../../../../scripts/chromaseed_weight_average_run.py#L28)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_weight_average_v1"
```

[Строка 29](../../../../scripts/chromaseed_weight_average_run.py#L29)

```python
SEEDS = (17, 29, 43)
```

[Строка 30](../../../../scripts/chromaseed_weight_average_run.py#L30)

```python
PARENT = "18b9127d82ca32e854be7efbf3906dd4f1fa73a749bf9d41ab7a78eae7df795b"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `freeze` | FunctionDef | См. реализацию | [L33](../../../../scripts/chromaseed_weight_average_run.py#L33) |
| `load_parents` | FunctionDef | См. реализацию | [L73](../../../../scripts/chromaseed_weight_average_run.py#L73) |
| `construct` | FunctionDef | См. реализацию | [L77](../../../../scripts/chromaseed_weight_average_run.py#L77) |
| `inner` | FunctionDef | См. реализацию | [L87](../../../../scripts/chromaseed_weight_average_run.py#L87) |
| `select` | FunctionDef | См. реализацию | [L118](../../../../scripts/chromaseed_weight_average_run.py#L118) |
| `evaluate` | FunctionDef | См. реализацию | [L164](../../../../scripts/chromaseed_weight_average_run.py#L164) |
| `main` | FunctionDef | См. реализацию | [L237](../../../../scripts/chromaseed_weight_average_run.py#L237) |
