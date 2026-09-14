# `scripts/cc_v7_external_controls.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_external_controls.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Additional frozen Fourier/classical source-only selectors for V7 transfer.

SHA-256 исходника: `ae86ac73499dc7d91e03d15794e94faf9cff74bf3bbedef6239db236ed8017cc`. Строк: **97**.

## Зависимости

```python
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cc_fourier_ridge import predict_score
from cc_v3_experiment import DATA_HASHES
from cc_v3_ffcc import decode, featurize
from cc_v7_risk import fit_heads, read_in_role_order
from cc_v7_verify import verify_manifest
from threadpoolctl import threadpool_limits
from torch.nn import functional as F
from luma_skin_vision.cc.core import EXPERT_NAMES, angular, reproduction
from luma_skin_vision.cc.v2 import risk_features_invariant
from luma_skin_vision.cc.v2_experiment import split_indices
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/cc_v7_external_controls.py#L22)

```python
ROOT=Path(__file__).resolve().parents[1]
```

[Строка 23](../../../../scripts/cc_v7_external_controls.py#L23)

```python
FILTER=ROOT/"experiments/runs/fourier_ridge_v1/sigma2.0_ridge0.001_gray_world.npz"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fourier_predictions` | FunctionDef | См. реализацию | [L27](../../../../scripts/cc_v7_external_controls.py#L27) |
| `classical_disagreement` | FunctionDef | См. реализацию | [L44](../../../../scripts/cc_v7_external_controls.py#L44) |
| `calibrated_raw` | FunctionDef | См. реализацию | [L48](../../../../scripts/cc_v7_external_controls.py#L48) |
| `run` | FunctionDef | См. реализацию | [L55](../../../../scripts/cc_v7_external_controls.py#L55) |
