# `scripts/chromaseed_weak_ridge_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_weak_ridge_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Expanded-ridge audit using frozen independent P reference algebra.

SHA-256 исходника: `757a6af83acee889f1c6419989aac5abc66fc51bde2bc2b524080bc10190e857`. Строк: **180**.

## Зависимости

```python
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
from chromaseed_kernel_audit import direct_kernel, independent_predict, js, norm, nz, unpack
from chromaseed_perceptual_audit import balanced, name_for, reference_readouts
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_weak_ridge_audit.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 15](../../../../scripts/chromaseed_weak_ridge_audit.py#L15)

```python
FAMILIES = ("norm_mse", "constant_de2", "local_de2", "local_irls", "midpoint_irls")
```

[Строка 16](../../../../scripts/chromaseed_weak_ridge_audit.py#L16)

```python
SEEDS = (17, 29, 43)
```

[Строка 17](../../../../scripts/chromaseed_weak_ridge_audit.py#L17)

```python
ALPHAS = (.0001, .0003, .001, .003, .01, .03, .1, 1., 10.)
```

[Строка 18](../../../../scripts/chromaseed_weak_ridge_audit.py#L18)

```python
WIDTHS = (.5, 1., 2.)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `audit` | FunctionDef | См. реализацию | [L21](../../../../scripts/chromaseed_weak_ridge_audit.py#L21) |
