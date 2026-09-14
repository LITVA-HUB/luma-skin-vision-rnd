# `scripts/chromaseed_lossless_arithmetic_probe.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_lossless_arithmetic_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Lossless modular integer differences of FP32 bit patterns; CPU probe only.

SHA-256 исходника: `0c5571dea9ad2be7d2bb22df3993a9cf741dee63f38bdf4114c7cd623ac53ab3`. Строк: **111**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import time
import zlib
from pathlib import Path
import numpy as np
import torch
from chromaseed_architecture_scale import VARIANTS, Bank, capacity
from chromaseed_architecture_scale_run import ROOT, bank_path
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_lossless_arithmetic_probe.py#L17)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_lossless_storage_probe"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `trial` | FunctionDef | См. реализацию | [L20](../../../../scripts/chromaseed_lossless_arithmetic_probe.py#L20) |
| `main` | FunctionDef | См. реализацию | [L45](../../../../scripts/chromaseed_lossless_arithmetic_probe.py#L45) |
