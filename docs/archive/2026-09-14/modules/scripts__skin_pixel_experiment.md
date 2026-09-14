# `scripts/skin_pixel_experiment.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_pixel_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed UCI skin-pixel classifier screen; not a skin-color accuracy benchmark.

SHA-256 исходника: `449f13914261f2085739936ec0caa01eed49b1df2c7e3657aa81d062b9c6ca22`. Строк: **112**.

## Зависимости

```python
from __future__ import annotations
import json
import pickle
import time
from pathlib import Path
import numpy as np
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `metrics` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_pixel_experiment.py#L18) |
| `choose` | FunctionDef | См. реализацию | [L34](../../../../scripts/skin_pixel_experiment.py#L34) |
| `main` | FunctionDef | См. реализацию | [L38](../../../../scripts/skin_pixel_experiment.py#L38) |
