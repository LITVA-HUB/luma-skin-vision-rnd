# `scripts/chromaseed_lossless_storage_probe.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_lossless_storage_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

CPU-only lossless-storage estimate from completed AS checkpoints.

This probe never rewrites an AS archive and is not a production checkpoint
format. Reconstructed bytes must be identical, including all FP32 bits.

SHA-256 исходника: `7b78397cee575d916e791b75f7ebc428f76b572d3167941e588c00b88dba2f46`. Строк: **138**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import shutil
import time
import zlib
from pathlib import Path
import numpy as np
import torch
from chromaseed_architecture_scale import VARIANTS, Bank, capacity
from chromaseed_architecture_scale_run import ROOT, RUN, bank_path
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/chromaseed_lossless_storage_probe.py#L22)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_lossless_storage_probe"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `trial` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_lossless_storage_probe.py#L25) |
| `main` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_lossless_storage_probe.py#L54) |
