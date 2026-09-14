# `scripts/cc_v7_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

CPU checkpoint/teacher replay and independent FP64 rescoring of V7 artifacts.

SHA-256 исходника: `2f614f674e5e838cb7ae1e8f419c182cad65a3f65208c2aa4666b2a5d1de0c06`. Строк: **138**.

## Зависимости

```python
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v7_core import SemanticColorNet, teacher_render
from cc_v7_experiment import evaluate
from torch.nn import functional as F
from luma_skin_vision.cc.v2 import CompactResidualCC
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/cc_v7_verify.py#L22)

```python
ROOT=Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `verify_manifest` | FunctionDef | См. реализацию | [L25](../../../../scripts/cc_v7_verify.py#L25) |
| `score` | FunctionDef | См. реализацию | [L32](../../../../scripts/cc_v7_verify.py#L32) |
| `teacher_replay` | FunctionDef | См. реализацию | [L38](../../../../scripts/cc_v7_verify.py#L38) |
| `run` | FunctionDef | См. реализацию | [L75](../../../../scripts/cc_v7_verify.py#L75) |
