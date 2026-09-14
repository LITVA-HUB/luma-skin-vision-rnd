# `scripts/chromaseed_architecture_scale_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_architecture_scale_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent NumPy export, lineage and metric audit for the AS screen.

SHA-256 исходника: `b157f54193a4991c0fdbaac853f88b697637e851aa319ec16008adce7c185343`. Строк: **322**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import time
import numpy as np
from chromaseed_architecture_scale import Bank, Predictor, base_model, predict
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, WE, bank_path, check_map, load_data
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, verify_normalizers
from chromaseed_neural_prefix_numpy import predict as warm_predict
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from chromaseed_widen_audit import exact
from chromaseed_widen_run import bank_path as old_bank
from skin_local_search_train import folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/chromaseed_architecture_scale_audit.py#L22)

```python
PARAMETERS = dict(
    patch_small=17374,
    patch5m=4962566,
    soft_small=15246,
    soft5m=4846822,
    dynamic_small=15246,
    dynamic5m=4846822,
    pool5m=4851846,
)
```

[Строка 31](../../../../scripts/chromaseed_architecture_scale_audit.py#L31)

```python
SEEDS = (17, 29, 43)
```

[Строка 32](../../../../scripts/chromaseed_architecture_scale_audit.py#L32)

```python
RATES = (0.00001, 0.0001)
```

[Строка 33](../../../../scripts/chromaseed_architecture_scale_audit.py#L33)

```python
TIMES = (128, 512, 2048)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `rank` | FunctionDef | См. реализацию | [L36](../../../../scripts/chromaseed_architecture_scale_audit.py#L36) |
| `compare` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_architecture_scale_audit.py#L43) |
| `inspect` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_architecture_scale_audit.py#L49) |
| `check_model` | FunctionDef | См. реализацию | [L103](../../../../scripts/chromaseed_architecture_scale_audit.py#L103) |
| `main` | FunctionDef | См. реализацию | [L118](../../../../scripts/chromaseed_architecture_scale_audit.py#L118) |
