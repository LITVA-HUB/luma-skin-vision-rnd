# `scripts/skin_teacher_features.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_teacher_features.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Immutable source-only DINOv2 skin features; no online loading or fitting.

SHA-256 исходника: `f5662d0ed61eb79d4c5b1eecce39555bf7e0658975ec6cb4fb9a8ed99f5cb447`. Строк: **118**.

## Зависимости

```python
import os
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from skin_mskcc_data import ROOT, sha
from skin_mskcc_pixels import load as load_pixels
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/skin_teacher_features.py#L16)

```python
OUT = ROOT/'docs/benchmarks/skin_teacher_readout_v1'
```

[Строка 17](../../../../scripts/skin_teacher_features.py#L17)

```python
CACHE = ROOT/'data/processed/skin_teacher_readout_v1'
```

[Строка 18](../../../../scripts/skin_teacher_features.py#L18)

```python
PROTOCOL = ROOT/'docs/research/skin_teacher_readout_protocol_v1.md'
```

[Строка 19](../../../../scripts/skin_teacher_features.py#L19)

```python
TEACHER = ROOT/'artifacts/teachers/dinov2-7764ea0f912e'
```

[Строка 20](../../../../scripts/skin_teacher_features.py#L20)

```python
WEIGHT_SHA = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `preprocess` | FunctionDef | См. реализацию | [L23](../../../../scripts/skin_teacher_features.py#L23) |
| `teacher` | FunctionDef | См. реализацию | [L33](../../../../scripts/skin_teacher_features.py#L33) |
| `extract` | FunctionDef | См. реализацию | [L49](../../../../scripts/skin_teacher_features.py#L49) |
| `load` | FunctionDef | См. реализацию | [L62](../../../../scripts/skin_teacher_features.py#L62) |
| `main` | FunctionDef | См. реализацию | [L76](../../../../scripts/skin_teacher_features.py#L76) |
