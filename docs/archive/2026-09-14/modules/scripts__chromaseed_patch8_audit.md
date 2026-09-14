# `scripts/chromaseed_patch8_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_patch8_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent P8 lineage, local reductions, selection and actual consumers.

SHA-256 исходника: `4cff72d955ebbe041f6e2fb10657759e882bbdb2b06571d9c8e1c65be5f63edd`. Строк: **138**.

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
from chromaseed_kernel_audit import js,nz
from chromaseed_local_denoise_audit import close,full_metrics,verify_normalizers
from chromaseed_long_training_run import ND,NP
from chromaseed_neural_prefix_audit import inspect_export
from chromaseed_patch8_numpy import Predictor
from chromaseed_patch8_run import OUT,ROOT,RUN,bank_path,check_map,load_data
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from skin_local_search_train import folds_for,roles,sha,write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 25](../../../../scripts/chromaseed_patch8_audit.py#L25)

```python
SEEDS=(17,29,43)
```

[Строка 26](../../../../scripts/chromaseed_patch8_audit.py#L26)

```python
TIMES=(0,512,2048,8192,32768)
```

[Строка 27](../../../../scripts/chromaseed_patch8_audit.py#L27)

```python
RATES=(.0001,.0003,.001)
```

[Строка 28](../../../../scripts/chromaseed_patch8_audit.py#L28)

```python
ARMS=("stats","patch8")
```

[Строка 29](../../../../scripts/chromaseed_patch8_audit.py#L29)

```python
SLOTS=[dict(seed=s,arm=a,lr=r) for s in SEEDS for a in ARMS for r in RATES]
```

[Строка 30](../../../../scripts/chromaseed_patch8_audit.py#L30)

```python
EXTRA={"u","e","g","t_mean","t_std"}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `direct` | FunctionDef | См. реализацию | [L33](../../../../scripts/chromaseed_patch8_audit.py#L33) |
| `transform_local` | FunctionDef | См. реализацию | [L47](../../../../scripts/chromaseed_patch8_audit.py#L47) |
| `gpu_parity` | FunctionDef | См. реализацию | [L57](../../../../scripts/chromaseed_patch8_audit.py#L57) |
| `exact` | FunctionDef | См. реализацию | [L77](../../../../scripts/chromaseed_patch8_audit.py#L77) |
| `inspect_bank` | FunctionDef | См. реализацию | [L83](../../../../scripts/chromaseed_patch8_audit.py#L83) |
