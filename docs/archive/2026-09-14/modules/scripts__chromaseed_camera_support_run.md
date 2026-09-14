# `scripts/chromaseed_camera_support_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_camera_support_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fit registered camera classifiers and measure input/target coverage, TRAIN only.

SHA-256 исходника: `273324e47e6c81ca0ab07677f35548f245b39426a0f5fce664d1e0f1b2ffbd28`. Строк: **257**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import scipy
from chromaseed_camera_support import (
    CALIPERS,
    METHODS,
    VIEWS,
    aggregate_people,
    classification_metrics,
    classify,
    describe,
    match_cost,
    nearest,
    row_weights,
    standardize,
    views,
    weighted_quantile,
)
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 32](../../../../scripts/chromaseed_camera_support_run.py#L32)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L35](../../../../scripts/chromaseed_camera_support_run.py#L35) |
| `lock_sources` | FunctionDef | См. реализацию | [L39](../../../../scripts/chromaseed_camera_support_run.py#L39) |
| `main` | FunctionDef | См. реализацию | [L82](../../../../scripts/chromaseed_camera_support_run.py#L82) |
