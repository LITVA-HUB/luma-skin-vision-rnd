# `scripts/chromaseed_neural_shrinkage_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_shrinkage_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent checks of every saved NS model, choice and actual consumer.

SHA-256 исходника: `292d22185e1f259baa48c3756e173fb097c4f7caf8f38322a423c7050bf64ad5`. Строк: **366**.

## Зависимости

```python
from __future__ import annotations
import argparse
import itertools
import time
from collections import Counter
from pathlib import Path
import numpy as np
from chromaseed_affine_audit import exact, summaries
from chromaseed_feature_groups_audit import compare
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS, actual_consumer, direct, row_hash, scoring
from chromaseed_kernel_audit import js, nz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 21](../../../../scripts/chromaseed_neural_shrinkage_audit.py#L21)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 22](../../../../scripts/chromaseed_neural_shrinkage_audit.py#L22)

```python
RUN = ROOT / "experiments/runs/chromaseed_neural_shrinkage_v1"
```

[Строка 23](../../../../scripts/chromaseed_neural_shrinkage_audit.py#L23)

```python
NR = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
```

[Строка 24](../../../../scripts/chromaseed_neural_shrinkage_audit.py#L24)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_neural_shrinkage_v1"
```

[Строка 25](../../../../scripts/chromaseed_neural_shrinkage_audit.py#L25)

```python
BASES = (
    "random",
    "adam_e1",
    "adam_e4",
    "adam_e16",
    "tagi_full3_e1",
    "tagi_full3_e4",
    "tagi_full3_e16",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ident` | FunctionDef | См. реализацию | [L42](../../../../scripts/chromaseed_neural_shrinkage_audit.py#L42) |
| `main` | FunctionDef | См. реализацию | [L46](../../../../scripts/chromaseed_neural_shrinkage_audit.py#L46) |
