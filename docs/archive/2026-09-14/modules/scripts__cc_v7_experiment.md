# `scripts/cc_v7_experiment.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched source-only compact semantic/sensor experiment; teacher is train-only.

SHA-256 исходника: `0499a8caddc851fcf0cc55371cfb91260be83a4894de389a198dd7ea70c7c2bd`. Строк: **240**.

## Зависимости

```python
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v5_experiment import capture_state, restore_state, state_digest
from cc_v7_core import SemanticColorNet, sensor_matrix, sensor_transform
from cc_v7_teacher import training_indices
from torch.nn import functional as F
from luma_skin_vision.cc.core import angular, reproduction, summarize
from luma_skin_vision.cc.v2_experiment import source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 26](../../../../scripts/cc_v7_experiment.py#L26)

```python
ROOT=Path(__file__).resolve().parents[1]
```

[Строка 27](../../../../scripts/cc_v7_experiment.py#L27)

```python
ARMS=("gt_native","gt_sensor","raw_teacher_sensor","canonical_teacher_sensor","canonical_teacher_native")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `augment_batch` | FunctionDef | См. реализацию | [L30](../../../../scripts/cc_v7_experiment.py#L30) |
| `distillation_loss` | FunctionDef | См. реализацию | [L40](../../../../scripts/cc_v7_experiment.py#L40) |
| `point_loss` | FunctionDef | См. реализацию | [L46](../../../../scripts/cc_v7_experiment.py#L46) |
| `load_teacher` | FunctionDef | См. реализацию | [L56](../../../../scripts/cc_v7_experiment.py#L56) |
| `train_epoch` | FunctionDef | См. реализацию | [L84](../../../../scripts/cc_v7_experiment.py#L84) |
| `evaluate` | FunctionDef | См. реализацию | [L118](../../../../scripts/cc_v7_experiment.py#L118) |
| `run` | FunctionDef | См. реализацию | [L133](../../../../scripts/cc_v7_experiment.py#L133) |
