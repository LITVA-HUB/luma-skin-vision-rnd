# `scripts/chromaseed_condensed_exact_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_condensed_exact_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Locked equivalence and paired runtime follow-up; no new model selection.

SHA-256 исходника: `2550195d8eb1cb9fc55a55fce305867959871762486ffc38302b5cc14f47219b`. Строк: **199**.

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
import torch
from chromaseed_condensed_exact import fit_condensed
from chromaseed_fast_kernel import fit_one, get_model, model_id
from chromaseed_fast_kernel_runtime import allocation_peak
from chromaseed_kernel import predict_kernel
from chromaseed_kernel_runtime import time_queries
from chromaseed_kernel_train import atomic_npz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/chromaseed_condensed_exact_run.py#L22)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_condensed_exact_run.py#L25) |
| `nz` | FunctionDef | См. реализацию | [L29](../../../../scripts/chromaseed_condensed_exact_run.py#L29) |
| `freeze` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_condensed_exact_run.py#L34) |
| `execute` | FunctionDef | См. реализацию | [L59](../../../../scripts/chromaseed_condensed_exact_run.py#L59) |
| `main` | FunctionDef | См. реализацию | [L189](../../../../scripts/chromaseed_condensed_exact_run.py#L189) |
