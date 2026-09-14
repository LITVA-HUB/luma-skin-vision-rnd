# `scripts/chromaseed_gaussian_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gaussian_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent TG bank/OOF/curve audit and full scalar retraining of selected traces.

SHA-256 исходника: `6312a3d1bf4c0243c092db6f6f29cec2630d7b931ebf4b94a9d68e3e7750bf61`. Строк: **585**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import itertools
import time
from pathlib import Path
import numpy as np
from chromaseed_affine_audit import exact, summaries
from chromaseed_feature_groups_audit import compare
from chromaseed_feature_groups_audit import direct as fg_direct
from chromaseed_feature_groups_numpy import Predictor as FGPredictor
from chromaseed_gate_stability_audit import close, transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_numpy import Predictor
from chromaseed_gaussian_reference import normalization, train_reference
from chromaseed_gaussian_reference import predict as ref_predict
from chromaseed_kernel_audit import js, nz
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 28](../../../../scripts/chromaseed_gaussian_audit.py#L28)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 29](../../../../scripts/chromaseed_gaussian_audit.py#L29)

```python
FG = ROOT / "experiments/runs/chromaseed_feature_groups_v1"
```

[Строка 30](../../../../scripts/chromaseed_gaussian_audit.py#L30)

```python
SOURCE = "c5f2d9f4a4a9111b4da1a5b238c0d48ecd6c851a808efb272d66e5c92cebca62"
```

[Строка 31](../../../../scripts/chromaseed_gaussian_audit.py#L31)

```python
METHODS = ("adam", "tagi_diag", "tagi_full3")
```

[Строка 32](../../../../scripts/chromaseed_gaussian_audit.py#L32)

```python
GROUPS = ("raw36", "mean3")
```

[Строка 33](../../../../scripts/chromaseed_gaussian_audit.py#L33)

```python
SEEDS = (17, 29, 43)
```

[Строка 34](../../../../scripts/chromaseed_gaussian_audit.py#L34)

```python
EPOCHS = (1, 4, 16, 64)
```

[Строка 35](../../../../scripts/chromaseed_gaussian_audit.py#L35)

```python
PARAMETERS = {
    "adam": (0.0003, 0.001, 0.003),
    "tagi_diag": (0.1, 0.3, 1.0),
    "tagi_full3": (0.1, 0.3, 1.0),
}
```

[Строка 40](../../../../scripts/chromaseed_gaussian_audit.py#L40)

```python
SETTINGS = [(0.0, np.zeros(3))] + [
    (d / 255, np.array(a)) for d in (1, 4, 16, 64) for a in itertools.product((0.0, 1.0), repeat=3)
]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ident` | FunctionDef | См. реализацию | [L45](../../../../scripts/chromaseed_gaussian_audit.py#L45) |
| `row_hash` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_gaussian_audit.py#L49) |
| `direct` | FunctionDef | См. реализацию | [L53](../../../../scripts/chromaseed_gaussian_audit.py#L53) |
| `actual_consumer` | FunctionDef | См. реализацию | [L57](../../../../scripts/chromaseed_gaussian_audit.py#L57) |
| `scoring` | FunctionDef | См. реализацию | [L61](../../../../scripts/chromaseed_gaussian_audit.py#L61) |
| `expected_names` | FunctionDef | См. реализацию | [L69](../../../../scripts/chromaseed_gaussian_audit.py#L69) |
| `bank_check` | FunctionDef | См. реализацию | [L112](../../../../scripts/chromaseed_gaussian_audit.py#L112) |
| `main` | FunctionDef | См. реализацию | [L226](../../../../scripts/chromaseed_gaussian_audit.py#L226) |
