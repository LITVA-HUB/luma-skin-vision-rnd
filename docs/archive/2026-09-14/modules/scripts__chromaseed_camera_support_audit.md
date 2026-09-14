# `scripts/chromaseed_camera_support_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_camera_support_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent weights, SVD pipelines, exhaustive matches and support calculations.

SHA-256 исходника: `02da1dde3eb9879395166f6f9411655d4f9373856f5bb7ad8ba0cdde29a307a7`. Строк: **346**.

## Зависимости

```python
from __future__ import annotations
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/chromaseed_camera_support_audit.py#L18)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L21](../../../../scripts/chromaseed_camera_support_audit.py#L21) |
| `weights` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_camera_support_audit.py#L25) |
| `norm` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_camera_support_audit.py#L43) |
| `aggregate` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_camera_support_audit.py#L54) |
| `metric` | FunctionDef | См. реализацию | [L69](../../../../scripts/chromaseed_camera_support_audit.py#L69) |
| `close` | FunctionDef | См. реализацию | [L82](../../../../scripts/chromaseed_camera_support_audit.py#L82) |
| `quantile` | FunctionDef | См. реализацию | [L99](../../../../scripts/chromaseed_camera_support_audit.py#L99) |
| `describe` | FunctionDef | См. реализацию | [L108](../../../../scripts/chromaseed_camera_support_audit.py#L108) |
| `direct_dist` | FunctionDef | См. реализацию | [L116](../../../../scripts/chromaseed_camera_support_audit.py#L116) |
| `main` | FunctionDef | См. реализацию | [L122](../../../../scripts/chromaseed_camera_support_audit.py#L122) |
