# `scripts/cc_projector_probe.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_projector_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

TRAIN-only subspace invariance and GT-assisted positive color-cone diagnostics.

SHA-256 исходника: `635571aa10f036feab9e3e66537966ca9e9052e83fa24c89e23ac248e1176e09`. Строк: **140**.

## Зависимости

```python
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES
from cc_v7_teacher import training_indices
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits
from torch.nn import functional as F
from luma_skin_vision.cc.core import angular, reproduction, summarize, unit
from luma_skin_vision.cc.v2 import CompactResidualCC
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 21](../../../../scripts/cc_projector_probe.py#L21)

```python
ROOT=Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `projector_features` | FunctionDef | См. реализацию | [L24](../../../../scripts/cc_projector_probe.py#L24) |
| `cone_oracle` | FunctionDef | См. реализацию | [L40](../../../../scripts/cc_projector_probe.py#L40) |
| `run` | FunctionDef | См. реализацию | [L65](../../../../scripts/cc_projector_probe.py#L65) |
