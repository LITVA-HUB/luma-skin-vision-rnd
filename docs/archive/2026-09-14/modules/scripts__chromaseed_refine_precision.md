# `scripts/chromaseed_refine_precision.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_refine_precision.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Storage-only encodings; inference remains decoded float32.

SHA-256 исходника: `56df57f9f58c42370739124e3cfa1d9aa92e4e2a408a3df8d83872d79ca061e2`. Строк: **132**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from chromaseed_refine_audit import error_summary, read_npz
from chromaseed_refine_numpy import NumpyRefiner
from chromaseed_refine_train import atomic_npz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/chromaseed_refine_precision.py#L12)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `encode` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_refine_precision.py#L23) |
| `decode` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_refine_precision.py#L43) |
| `evaluate` | FunctionDef | См. реализацию | [L61](../../../../scripts/chromaseed_refine_precision.py#L61) |
| `main` | FunctionDef | См. реализацию | [L121](../../../../scripts/chromaseed_refine_precision.py#L121) |
