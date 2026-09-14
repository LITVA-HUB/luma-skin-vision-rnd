# `scripts/chromaseed_fast_kernel_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_fast_kernel_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent width/role/decision/SVD/prediction checks for the frozen KF run.

SHA-256 исходника: `86bf04413ebe7895a19ab69ef92788f5a21dd7ca3c1fa720e54a186cebd53627`. Строк: **253**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from chromaseed_kernel_audit import direct_kernel, independent_predict, norm, unpack
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_fast_kernel_audit.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L17](../../../../scripts/chromaseed_fast_kernel_audit.py#L17) |
| `nz` | FunctionDef | См. реализацию | [L21](../../../../scripts/chromaseed_fast_kernel_audit.py#L21) |
| `key_for` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_fast_kernel_audit.py#L26) |
| `row_hash` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_fast_kernel_audit.py#L30) |
| `positive_median` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_fast_kernel_audit.py#L34) |
| `independent_widths` | FunctionDef | См. реализацию | [L39](../../../../scripts/chromaseed_fast_kernel_audit.py#L39) |
| `summarize` | FunctionDef | См. реализацию | [L58](../../../../scripts/chromaseed_fast_kernel_audit.py#L58) |
| `refit_svd` | FunctionDef | См. реализацию | [L63](../../../../scripts/chromaseed_fast_kernel_audit.py#L63) |
| `audit` | FunctionDef | См. реализацию | [L75](../../../../scripts/chromaseed_fast_kernel_audit.py#L75) |
| `main` | FunctionDef | См. реализацию | [L244](../../../../scripts/chromaseed_fast_kernel_audit.py#L244) |
