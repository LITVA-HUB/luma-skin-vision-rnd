# `scripts/chromaseed_neural_geometry_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_geometry_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen inner-only NR geometry and independent reconstruction.

SHA-256 исходника: `09cf5bb83bd57d8669eb985ff5c4047feaec7ddf2eaab329cd8c6c3ffdb362e6`. Строк: **248**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import numpy as np
from chromaseed_fast_kernel_train import row_hash
from chromaseed_gated import unpack
from chromaseed_gaussian_audit import scoring
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_neural_geometry import degrees, source_design, source_metric, spectrum
from chromaseed_neural_geometry_reference import reconstruct
from chromaseed_neural_readout import BASES, FAMILIES, GROUPS, SEEDS, name_for
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 21](../../../../scripts/chromaseed_neural_geometry_run.py#L21)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 22](../../../../scripts/chromaseed_neural_geometry_run.py#L22)

```python
PARENT = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
```

[Строка 23](../../../../scripts/chromaseed_neural_geometry_run.py#L23)

```python
RECEIPT = ROOT / "docs/benchmarks/chromaseed_neural_readout_v1/verification.json"
```

[Строка 24](../../../../scripts/chromaseed_neural_geometry_run.py#L24)

```python
RECEIPT_HASH = "3e4079df5d688136ef59be6b599633b0bc7c44d9d21f410eaeefc2aef0699c65"
```

[Строка 25](../../../../scripts/chromaseed_neural_geometry_run.py#L25)

```python
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `freeze` | FunctionDef | См. реализацию | [L28](../../../../scripts/chromaseed_neural_geometry_run.py#L28) |
| `main` | FunctionDef | См. реализацию | [L89](../../../../scripts/chromaseed_neural_geometry_run.py#L89) |
