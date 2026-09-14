# `scripts/chromaseed_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Staged, hash-bound palette pretraining and original-TRAIN skin adaptation.

SHA-256 исходника: `82c215dfb9180e37e77365feab4a3a0df0d4842179124cddd3eaec4628596baf`. Строк: **386**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from chromaseed import (
    X_MEAN,
    X_STD,
    Y_MEAN,
    Y_STD,
    ChromaSeed,
    make_palette,
    pack_model,
    predict,
    unpack_model,
)
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    load_model,
    metrics,
    roles,
    sha,
    synchronize,
    weights_for,
    write_json,
)
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 38](../../../../scripts/chromaseed_train.py#L38)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 39](../../../../scripts/chromaseed_train.py#L39)

```python
SEEDS = (17, 29, 43)
```

[Строка 40](../../../../scripts/chromaseed_train.py#L40)

```python
ARMS = ("scratch", "clean_palette", "rendered_palette", "shuffled_palette", "skin_long")
```

[Строка 41](../../../../scripts/chromaseed_train.py#L41)

```python
PRETRAINED = ("clean_palette", "rendered_palette", "shuffled_palette")
```

[Строка 42](../../../../scripts/chromaseed_train.py#L42)

```python
LRS = (.0003, .001, .003)
```

[Строка 43](../../../../scripts/chromaseed_train.py#L43)

```python
CHECKPOINTS = (64, 256, 1024)
```

[Строка 44](../../../../scripts/chromaseed_train.py#L44)

```python
PRETRAIN_STEPS = 2048
```

[Строка 45](../../../../scripts/chromaseed_train.py#L45)

```python
PROTOCOL = ROOT / "docs/research/chromaseed_v1_protocol.md"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `train_trace` | FunctionDef | См. реализацию | [L48](../../../../scripts/chromaseed_train.py#L48) |
| `write_trace` | FunctionDef | См. реализацию | [L91](../../../../scripts/chromaseed_train.py#L91) |
| `read_trace` | FunctionDef | См. реализацию | [L104](../../../../scripts/chromaseed_train.py#L104) |
| `obtain_trace` | FunctionDef | См. реализацию | [L115](../../../../scripts/chromaseed_train.py#L115) |
| `palette_metrics` | FunctionDef | См. реализацию | [L125](../../../../scripts/chromaseed_train.py#L125) |
| `lock_sources` | FunctionDef | См. реализацию | [L131](../../../../scripts/chromaseed_train.py#L131) |
| `prepare_palette` | FunctionDef | См. реализацию | [L153](../../../../scripts/chromaseed_train.py#L153) |
| `pretrain_phase` | FunctionDef | См. реализацию | [L176](../../../../scripts/chromaseed_train.py#L176) |
| `verify_pretraining` | FunctionDef | См. реализацию | [L214](../../../../scripts/chromaseed_train.py#L214) |
| `adaptation_initial` | FunctionDef | См. реализацию | [L229](../../../../scripts/chromaseed_train.py#L229) |
| `fit_phase` | FunctionDef | См. реализацию | [L241](../../../../scripts/chromaseed_train.py#L241) |
| `evaluate_phase` | FunctionDef | См. реализацию | [L308](../../../../scripts/chromaseed_train.py#L308) |
| `main` | FunctionDef | См. реализацию | [L357](../../../../scripts/chromaseed_train.py#L357) |
