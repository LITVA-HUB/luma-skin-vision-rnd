# `docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Paired, deterministic source-development correction-critic experiment.

V4 executable bytes remain unchanged. See docs/research/cc_v5_protocol.md.

SHA-256 исходника: `5bd8300b92743ef4da235ccb1755956bd394d5443f644fe9e7e1cbb0164635d2`. Строк: **266**.

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
from cc_v4_experiment import evaluate, field_weight
from cc_v5_model import CorrectionEvidenceNet, analytic_costs
from luma_skin_vision.cc.v2_experiment import source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 28](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py#L28)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 29](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py#L29)

```python
ARMS = ('point', 'posterior_random', 'action_random', 'transport_random',
        'transport_policy', 'transport_gradient')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `capture_state` | FunctionDef | См. реализацию | [L33](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py#L33) |
| `restore_state` | FunctionDef | См. реализацию | [L38](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py#L38) |
| `state_digest` | FunctionDef | См. реализацию | [L45](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py#L45) |
| `draw_actions` | FunctionDef | См. реализацию | [L67](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py#L67) |
| `physical_targets` | FunctionDef | См. реализацию | [L74](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py#L74) |
| `train_epoch` | FunctionDef | См. реализацию | [L82](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py#L82) |
| `run` | FunctionDef | См. реализацию | [L128](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed29/source_snapshot/scripts/cc_v5_experiment.py#L128) |
