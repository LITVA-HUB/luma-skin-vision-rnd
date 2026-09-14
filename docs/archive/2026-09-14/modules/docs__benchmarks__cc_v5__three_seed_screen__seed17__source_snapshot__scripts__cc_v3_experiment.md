# `docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Immutable source-only V3 graph/posterior screen. No test rows are decoded.

SHA-256 исходника: `2270c8a05706f685eaeb4558b58b64d94ddebbfa65e6301b5c5dbe9b97d0b5fc`. Строк: **283**.

## Зависимости

```python
import argparse
import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_model import ColorFramePosteriorNet, camera_posterior_nll, posterior_nll
from torch.nn import functional as F
from luma_skin_vision.cc.core import angular, reproduction, summarize
from luma_skin_vision.cc.v2_experiment import source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 23](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py#L23)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 24](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py#L24)

```python
DATA_HASHES = {
    "cube.npz": "8323048ad50deb5aa7a0f5c8b9787ed311e4c8779a8ff2d87160068a07e92128",
    "cube_manifest.json": "912927e16f32a55b0289910f32790be0ddaf34062ce71a1bf3fb98ae75a7edee",
}
```

[Строка 28](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py#L28)

```python
LOSS = {"reproduction_degrees": 1.0, "canonical_nll": 0.02, "positivity": 10.0}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `source_rows` | FunctionDef | См. реализацию | [L31](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py#L31) |
| `point_objective` | FunctionDef | All-row clipped raw point surrogate plus normalized negativity penalty.  Evaluation uses explicit valid/fallback output. Clipping here only supplies a finite training surrogate; the separate penalty gives negative channels a repair gradient. It never turns an invalid output into an accepted one. | [L51](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py#L51) |
| `validation_key` | FunctionDef | См. реализацию | [L67](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py#L67) |
| `training_summary` | FunctionDef | См. реализацию | [L71](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py#L71) |
| `evaluate` | FunctionDef | См. реализацию | [L79](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py#L79) |
| `train` | FunctionDef | См. реализацию | [L109](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py#L109) |
| `main` | FunctionDef | См. реализацию | [L257](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_experiment.py#L257) |
