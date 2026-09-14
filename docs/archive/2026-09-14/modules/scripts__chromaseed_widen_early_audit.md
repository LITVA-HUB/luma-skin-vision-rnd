# `scripts/chromaseed_widen_early_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_widen_early_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent WE sampling/lineage, canaries, candidate reconstruction and consumers.

SHA-256 исходника: `277d361627beb6155d060f5c2a776ade09205b9207743db754ae138c94148b3a`. Строк: **462**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import time
import numpy as np
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, full_metrics, verify_normalizers
from chromaseed_neural_prefix_numpy import Predictor as BasePredictor
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from chromaseed_widen import Predictor
from chromaseed_widen_audit import PARAMS, direct, exact, independent_gpu, local_transform
from chromaseed_widen_early_run import (
    OUT,
    ROOT,
    RUN,
    WIDE,
    bank_path,
    check_map,
    load_data,
    wide_bank,
)
from skin_local_search_train import folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 33](../../../../scripts/chromaseed_widen_early_audit.py#L33)

```python
CAPS = ("m31", "m61", "m111", "m832")
```

[Строка 34](../../../../scripts/chromaseed_widen_early_audit.py#L34)

```python
FIRST = (0.00001, 0.00003, 0.0001)
```

[Строка 35](../../../../scripts/chromaseed_widen_early_audit.py#L35)

```python
SEEDS = (17, 29, 43)
```

[Строка 36](../../../../scripts/chromaseed_widen_early_audit.py#L36)

```python
TIMES = (0, 32, 128, 512, 2048)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `inspect_bank` | FunctionDef | См. реализацию | [L39](../../../../scripts/chromaseed_widen_early_audit.py#L39) |
| `audit_inner` | FunctionDef | См. реализацию | [L104](../../../../scripts/chromaseed_widen_early_audit.py#L104) |
| `audit_final` | FunctionDef | См. реализацию | [L261](../../../../scripts/chromaseed_widen_early_audit.py#L261) |
| `main` | FunctionDef | См. реализацию | [L412](../../../../scripts/chromaseed_widen_early_audit.py#L412) |
