# `scripts/cc_v6_experiment.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v6_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Paired canonical-frame correction evidence development experiment.

V5 executable bytes remain unchanged. See docs/research/cc_v6_combination_protocol.md.

SHA-256 исходника: `858042697b3472115dfdfebc46b3d09d05b248ddd529dac4a7fb51360c7056cf`. Строк: **269**.

## Зависимости

```python
import argparse
import copy
import hashlib
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
from cc_v4_experiment import field_weight
from cc_v5_model import analytic_costs
from cc_v6_evaluate import evaluate
from cc_v6_model import CanonicalEvidenceNet
from luma_skin_vision.cc.v2_experiment import source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 30](../../../../scripts/cc_v6_experiment.py#L30)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 31](../../../../scripts/cc_v6_experiment.py#L31)

```python
ARMS = ('point', 'posterior_random', 'action_random', 'transport_random')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `capture_state` | FunctionDef | См. реализацию | [L34](../../../../scripts/cc_v6_experiment.py#L34) |
| `restore_state` | FunctionDef | См. реализацию | [L39](../../../../scripts/cc_v6_experiment.py#L39) |
| `state_digest` | FunctionDef | См. реализацию | [L46](../../../../scripts/cc_v6_experiment.py#L46) |
| `draw_actions` | FunctionDef | См. реализацию | [L68](../../../../scripts/cc_v6_experiment.py#L68) |
| `physical_targets` | FunctionDef | См. реализацию | [L75](../../../../scripts/cc_v6_experiment.py#L75) |
| `train_epoch` | FunctionDef | См. реализацию | [L83](../../../../scripts/cc_v6_experiment.py#L83) |
| `run` | FunctionDef | См. реализацию | [L130](../../../../scripts/cc_v6_experiment.py#L130) |
