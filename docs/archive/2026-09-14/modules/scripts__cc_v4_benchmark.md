# `scripts/cc_v4_benchmark.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v4_benchmark.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Measured batch-one cached/refined inference; no export or test-set access.

SHA-256 исходника: `9f338d9dcd901a3d3d2c73cc86dfba142054ae71f4c9eca6df49d01ef2065a75`. Строк: **110**.

## Зависимости

```python
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v4_model import CorrectionEvidenceNet
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/cc_v4_benchmark.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L19](../../../../scripts/cc_v4_benchmark.py#L19) |
