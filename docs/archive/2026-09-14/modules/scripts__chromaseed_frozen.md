# `scripts/chromaseed_frozen.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_frozen.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen palette representation with an analytic, compact skin readout.

SHA-256 исходника: `3af2b628e6f6f3b6e56499c96e4268a64b136991434e8dcb9fdaf4be0cedaf17`. Строк: **176**.

## Зависимости

```python
import argparse
import json
import time
from pathlib import Path
import numpy as np
from chromaseed import X_MEAN, X_STD, Y_MEAN, Y_STD, predict
from chromaseed_train import PRETRAIN_STEPS, read_trace, verify_pretraining
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    load_model,
    metrics,
    roles,
    sha,
    weights_for,
    write_json,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 21](../../../../scripts/chromaseed_frozen.py#L21)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 22](../../../../scripts/chromaseed_frozen.py#L22)

```python
BASES = ("random_basis", "clean_palette", "rendered_palette", "shuffled_palette")
```

[Строка 23](../../../../scripts/chromaseed_frozen.py#L23)

```python
SEEDS = (17, 29, 43)
```

[Строка 24](../../../../scripts/chromaseed_frozen.py#L24)

```python
ALPHAS = (.1, 1., 10.)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fit_readout` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_frozen.py#L27) |
| `source_basis` | FunctionDef | См. реализацию | [L57](../../../../scripts/chromaseed_frozen.py#L57) |
| `lock` | FunctionDef | См. реализацию | [L64](../../../../scripts/chromaseed_frozen.py#L64) |
| `fit_phase` | FunctionDef | См. реализацию | [L83](../../../../scripts/chromaseed_frozen.py#L83) |
| `evaluate_phase` | FunctionDef | См. реализацию | [L130](../../../../scripts/chromaseed_frozen.py#L130) |
| `main` | FunctionDef | См. реализацию | [L155](../../../../scripts/chromaseed_frozen.py#L155) |
