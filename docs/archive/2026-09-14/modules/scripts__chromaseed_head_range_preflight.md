# `scripts/chromaseed_head_range_preflight.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_head_range_preflight.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Synthetic CUDA gate for HR; never loads the real TRAIN or held arrays.

SHA-256 исходника: `0d05ac6569b9611e59ade86003582fc992324af95e099339e6da2c27e246c623`. Строк: **96**.

## Зависимости

```python
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from chromaseed_architecture_scale import VARIANTS, capacity
from chromaseed_architecture_scale import fit as original_fit
from chromaseed_head_range import Predictor, predict_torch
from chromaseed_head_range_fit import fit
from chromaseed_refine_train import setup
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_head_range_preflight.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_head_range_preflight.py#L18)

```python
RUN = ROOT / 'experiments/runs/chromaseed_head_range_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `payload_digest` | FunctionDef | См. реализацию | [L21](../../../../scripts/chromaseed_head_range_preflight.py#L21) |
| `main` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_head_range_preflight.py#L26) |
