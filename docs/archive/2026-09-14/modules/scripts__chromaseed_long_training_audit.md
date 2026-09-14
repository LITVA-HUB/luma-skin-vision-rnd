# `scripts/chromaseed_long_training_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_long_training_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent LT data/lineage/selection/prediction and actual-consumer audit.

SHA-256 исходника: `f3488e623ea00b0929d7116d9258f703edf9a4004ca547344cb588d6189019d6`. Строк: **376**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import time
import numpy as np
import torch
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, full_metrics, verify_normalizers
from chromaseed_long_training_run import ND, NP, OUT, ROOT, RUN, bank_path, check_map, load_data
from chromaseed_neural_prefix_audit import inspect_export
from chromaseed_neural_prefix_numpy import Predictor
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from skin_local_search_train import folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 24](../../../../scripts/chromaseed_long_training_audit.py#L24)

```python
SEEDS = (17, 29, 43)
```

[Строка 25](../../../../scripts/chromaseed_long_training_audit.py#L25)

```python
STEPS = (0, 512, 2048, 8192, 32768, 131072)
```

[Строка 26](../../../../scripts/chromaseed_long_training_audit.py#L26)

```python
MODES = (0, 16, 256)
```

[Строка 27](../../../../scripts/chromaseed_long_training_audit.py#L27)

```python
RATES = (0.0001, 0.0003)
```

[Строка 28](../../../../scripts/chromaseed_long_training_audit.py#L28)

```python
SLOTS = [dict(seed=s, variants=v, lr=r) for s in SEEDS for v in MODES for r in RATES]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `direct` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_long_training_audit.py#L31) |
| `framework` | FunctionDef | См. реализацию | [L40](../../../../scripts/chromaseed_long_training_audit.py#L40) |
| `validate_data` | FunctionDef | См. реализацию | [L58](../../../../scripts/chromaseed_long_training_audit.py#L58) |
| `bank` | FunctionDef | См. реализацию | [L111](../../../../scripts/chromaseed_long_training_audit.py#L111) |
| `inner` | FunctionDef | См. реализацию | [L173](../../../../scripts/chromaseed_long_training_audit.py#L173) |
| `final` | FunctionDef | См. реализацию | [L245](../../../../scripts/chromaseed_long_training_audit.py#L245) |
| `main` | FunctionDef | См. реализацию | [L330](../../../../scripts/chromaseed_long_training_audit.py#L330) |
