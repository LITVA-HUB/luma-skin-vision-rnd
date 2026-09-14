# `src/luma_skin_vision/cc/benchmark.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/cc/benchmark.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `dee91bae44d06d64fd22173def745de9616eac6f1e5bb4ab24faa424ebf16dea`. Строк: **346**.

## Зависимости

```python
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.experiment import source_identity, write_json
from .core import EXPERT_NAMES, angular, reproduction, selective_curve, summarize
from .model import CompactCC, reproduction_loss
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `data_hashes` | FunctionDef | См. реализацию | [L16](../../../../src/luma_skin_vision/cc/benchmark.py#L16) |
| `load` | FunctionDef | См. реализацию | [L27](../../../../src/luma_skin_vision/cc/benchmark.py#L27) |
| `indices` | FunctionDef | См. реализацию | [L33](../../../../src/luma_skin_vision/cc/benchmark.py#L33) |
| `predict` | FunctionDef | См. реализацию | [L63](../../../../src/luma_skin_vision/cc/benchmark.py#L63) |
| `train` | FunctionDef | См. реализацию | [L73](../../../../src/luma_skin_vision/cc/benchmark.py#L73) |
| `risk_features` | FunctionDef | См. реализацию | [L163](../../../../src/luma_skin_vision/cc/benchmark.py#L163) |
| `fit_risk` | FunctionDef | См. реализацию | [L179](../../../../src/luma_skin_vision/cc/benchmark.py#L179) |
| `apply_risk` | FunctionDef | См. реализацию | [L197](../../../../src/luma_skin_vision/cc/benchmark.py#L197) |
| `result` | FunctionDef | См. реализацию | [L206](../../../../src/luma_skin_vision/cc/benchmark.py#L206) |
| `evaluate` | FunctionDef | См. реализацию | [L231](../../../../src/luma_skin_vision/cc/benchmark.py#L231) |
| `main` | FunctionDef | См. реализацию | [L331](../../../../src/luma_skin_vision/cc/benchmark.py#L331) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L63–70</summary>

```python
def predict(model, x, ex, batch=64):
    model.eval()
    pred, features = [], []
    for start in range(0, len(x), batch):
        p, f = model(x[start : start + batch], ex[start : start + batch])
        pred.append(p.cpu().numpy())
        features.append(f.cpu().numpy())
    return np.concatenate(pred), np.concatenate(features)
```

</details>
