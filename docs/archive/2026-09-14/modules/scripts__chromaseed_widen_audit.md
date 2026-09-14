# `scripts/chromaseed_widen_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_widen_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent capacity-study audit; never trains or changes frozen weights.

SHA-256 исходника: `2d656e246289dbb57df360d0dc7a1a8eb350b9cc9483195d79eaaa917257d173`. Строк: **481**.

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
from chromaseed_long_training_run import ND, NP
from chromaseed_neural_prefix_audit import inspect_export
from chromaseed_neural_prefix_numpy import Predictor as BasePredictor
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from chromaseed_widen import Predictor
from chromaseed_widen_run import OUT, ROOT, RUN, bank_path, check_map, load_data
from skin_local_search_train import folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 26](../../../../scripts/chromaseed_widen_audit.py#L26)

```python
SPECS = {
    "tiny": (8, 16),
    "m31": (64, 128),
    "m61": (64, 256),
    "m111": (128, 256),
    "m832": (256, 1024),
}
```

[Строка 33](../../../../scripts/chromaseed_widen_audit.py#L33)

```python
PARAMS = {"tiny": 1179, "m31": 30915, "m61": 60611, "m111": 110979, "m832": 832259}
```

[Строка 34](../../../../scripts/chromaseed_widen_audit.py#L34)

```python
SEEDS = (17, 29, 43)
```

[Строка 35](../../../../scripts/chromaseed_widen_audit.py#L35)

```python
RATES = (0.0001, 0.001)
```

[Строка 36](../../../../scripts/chromaseed_widen_audit.py#L36)

```python
TIMES = (0, 512, 2048, 8192)
```

[Строка 37](../../../../scripts/chromaseed_widen_audit.py#L37)

```python
SLOTS = [dict(seed=s, lr=r) for s in SEEDS for r in RATES]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `direct` | FunctionDef | Non-BLAS explicit contraction; independent from deployed matmul path. | [L40](../../../../scripts/chromaseed_widen_audit.py#L40) |
| `local_transform` | FunctionDef | См. реализацию | [L67](../../../../scripts/chromaseed_widen_audit.py#L67) |
| `exact` | FunctionDef | См. реализацию | [L74](../../../../scripts/chromaseed_widen_audit.py#L74) |
| `independent_gpu` | FunctionDef | См. реализацию | [L80](../../../../scripts/chromaseed_widen_audit.py#L80) |
| `inspect_bank` | FunctionDef | См. реализацию | [L110](../../../../scripts/chromaseed_widen_audit.py#L110) |
| `audit_inner` | FunctionDef | См. реализацию | [L204](../../../../scripts/chromaseed_widen_audit.py#L204) |
| `audit_final` | FunctionDef | См. реализацию | [L325](../../../../scripts/chromaseed_widen_audit.py#L325) |
| `main` | FunctionDef | См. реализацию | [L440](../../../../scripts/chromaseed_widen_audit.py#L440) |
