# `scripts/cc_v4_experiment.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v4_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-development evidence routing screen; never reads test/camera GT rows.

SHA-256 исходника: `d0c7aaf08fedc98bb2db37598eff730713dcd3a8c8e3e4f41fc1a00368de6eac`. Строк: **292**.

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
from cc_v3_experiment import DATA_HASHES, source_rows
from luma_skin_vision.cc.core import angular, reproduction, summarize
from luma_skin_vision.cc.v2_experiment import source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/cc_v4_experiment.py#L22)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 23](../../../../scripts/cc_v4_experiment.py#L23)

```python
LOSS_WEIGHTS = {"point_degrees": 1., "angular_mse_degrees": .05, "sin2_mse": 25.}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sample_actions` | FunctionDef | 33 actions; sampling sees no GT. Proposal is detached, RNG independent. | [L26](../../../../scripts/cc_v4_experiment.py#L26) |
| `field_weight` | FunctionDef | См. реализацию | [L34](../../../../scripts/cc_v4_experiment.py#L34) |
| `action_rgb` | FunctionDef | См. реализацию | [L38](../../../../scripts/cc_v4_experiment.py#L38) |
| `risk_summary` | FunctionDef | См. реализацию | [L44](../../../../scripts/cc_v4_experiment.py#L44) |
| `refinement_summary` | FunctionDef | См. реализацию | [L60](../../../../scripts/cc_v4_experiment.py#L60) |
| `oracle_errors` | FunctionDef | Oracle diagnostic over the actual policy's 103 evaluated candidates. | [L73](../../../../scripts/cc_v4_experiment.py#L73) |
| `evaluate` | FunctionDef | См. реализацию | [L90](../../../../scripts/cc_v4_experiment.py#L90) |
| `train` | FunctionDef | См. реализацию | [L128](../../../../scripts/cc_v4_experiment.py#L128) |
| `main` | FunctionDef | См. реализацию | [L270](../../../../scripts/cc_v4_experiment.py#L270) |
