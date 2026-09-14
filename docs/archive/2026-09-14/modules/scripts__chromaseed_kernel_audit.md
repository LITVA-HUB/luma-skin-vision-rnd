# `scripts/chromaseed_kernel_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_kernel_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Separate algebra/row/selection audit of the frozen ChromaSeed-K run.

SHA-256 исходника: `f68ec73e09176848c96b5e54fb6f9c3647d8b1a0fb67900f7a59547634e77e28`. Строк: **330**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json
from skin_local_search_train import predict as legacy_predict
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/chromaseed_kernel_audit.py#L12)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L20](../../../../scripts/chromaseed_kernel_audit.py#L20) |
| `nz` | FunctionDef | См. реализацию | [L24](../../../../scripts/chromaseed_kernel_audit.py#L24) |
| `key_for` | FunctionDef | См. реализацию | [L29](../../../../scripts/chromaseed_kernel_audit.py#L29) |
| `unpack` | FunctionDef | См. реализацию | [L33](../../../../scripts/chromaseed_kernel_audit.py#L33) |
| `norm` | FunctionDef | См. реализацию | [L38](../../../../scripts/chromaseed_kernel_audit.py#L38) |
| `direct_kernel` | FunctionDef | См. реализацию | [L42](../../../../scripts/chromaseed_kernel_audit.py#L42) |
| `independent_predict` | FunctionDef | См. реализацию | [L51](../../../../scripts/chromaseed_kernel_audit.py#L51) |
| `svd_refit` | FunctionDef | См. реализацию | [L55](../../../../scripts/chromaseed_kernel_audit.py#L55) |
| `summarize` | FunctionDef | См. реализацию | [L76](../../../../scripts/chromaseed_kernel_audit.py#L76) |
| `policy` | FunctionDef | См. реализацию | [L81](../../../../scripts/chromaseed_kernel_audit.py#L81) |
| `adaptive_trace` | FunctionDef | См. реализацию | [L93](../../../../scripts/chromaseed_kernel_audit.py#L93) |
| `main_audit` | FunctionDef | См. реализацию | [L114](../../../../scripts/chromaseed_kernel_audit.py#L114) |
