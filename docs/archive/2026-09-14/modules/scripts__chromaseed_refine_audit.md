# `scripts/chromaseed_refine_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_refine_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent row/selection/numerical audit and CPU timing of ChromaSeed-R.

SHA-256 исходника: `63ad91c019edfa64522fe30f01830308dd26a863ec45460b09cff4eef130c653`. Строк: **282**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from chromaseed_refine_numpy import NumpyRefiner
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_refine_audit.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read_json` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_refine_audit.py#L23) |
| `read_npz` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_refine_audit.py#L27) |
| `error_summary` | FunctionDef | См. реализацию | [L32](../../../../scripts/chromaseed_refine_audit.py#L32) |
| `independent_policy` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_refine_audit.py#L43) |
| `verify_preprocessor` | FunctionDef | См. реализацию | [L57](../../../../scripts/chromaseed_refine_audit.py#L57) |
| `check_metric` | FunctionDef | См. реализацию | [L73](../../../../scripts/chromaseed_refine_audit.py#L73) |
| `audit` | FunctionDef | См. реализацию | [L79](../../../../scripts/chromaseed_refine_audit.py#L79) |
| `runtime` | FunctionDef | См. реализацию | [L232](../../../../scripts/chromaseed_refine_audit.py#L232) |
| `main` | FunctionDef | См. реализацию | [L268](../../../../scripts/chromaseed_refine_audit.py#L268) |
