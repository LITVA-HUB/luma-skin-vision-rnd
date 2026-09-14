# `scripts/cc_v7_teacher.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_teacher.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Train-only frozen DINOv2 patch targets; no network access or inference teacher.

SHA-256 исходника: `70a99f8bcb9a7c0921c0baef8e1d914ddcd5a0343f32678e7a6f09c8700feb78`. Строк: **100**.

## Зависимости

```python
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES
from cc_v7_core import teacher_render
from torch.nn import functional as F
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 21](../../../../scripts/cc_v7_teacher.py#L21)

```python
ROOT=Path(__file__).resolve().parents[1]
```

[Строка 22](../../../../scripts/cc_v7_teacher.py#L22)

```python
WEIGHT_SHA="b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `training_indices` | FunctionDef | См. реализацию | [L25](../../../../scripts/cc_v7_teacher.py#L25) |
| `run` | FunctionDef | См. реализацию | [L38](../../../../scripts/cc_v7_teacher.py#L38) |
