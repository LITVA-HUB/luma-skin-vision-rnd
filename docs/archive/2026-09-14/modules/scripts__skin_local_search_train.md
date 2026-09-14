# `scripts/skin_local_search_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_search_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen TRAIN-only experiment. No legacy validation/test loaders are imported.

SHA-256 исходника: `c39e36191354c39bdbdf0d4ba7a073be27b844bc9ec9a24c0c954cfde55704bc`. Строк: **438**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from skin_local_search_core import (
    greedy_ridge_indices,
    kernel_ridge_fit,
    rbf_features,
    ridge_solve,
)
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/skin_local_search_train.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 28](../../../../scripts/skin_local_search_train.py#L28)

```python
CACHE_HASH = "d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0"
```

[Строка 29](../../../../scripts/skin_local_search_train.py#L29)

```python
ROLE_HASH = "7fa14adc41525a868d737539571c9d8cb9e954a680e457f828d7aab35a2bc064"
```

[Строка 30](../../../../scripts/skin_local_search_train.py#L30)

```python
METHODS = ("ridge", "krr", "random_rbf", "guided_rbf", "mlp")
```

[Строка 31](../../../../scripts/skin_local_search_train.py#L31)

```python
SEEDS = (17, 29, 43)
```

[Строка 32](../../../../scripts/skin_local_search_train.py#L32)

```python
GRIDS = {m: (0.1, 1.0, 10.0) for m in METHODS}
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `SmallMLP` | nn.Module | [L171](../../../../scripts/skin_local_search_train.py#L171) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_local_search_train.py#L36) |
| `write_json` | FunctionDef | См. реализацию | [L44](../../../../scripts/skin_local_search_train.py#L44) |
| `synchronize` | FunctionDef | См. реализацию | [L51](../../../../scripts/skin_local_search_train.py#L51) |
| `weights_for` | FunctionDef | См. реализацию | [L56](../../../../scripts/skin_local_search_train.py#L56) |
| `roles` | FunctionDef | См. реализацию | [L66](../../../../scripts/skin_local_search_train.py#L66) |
| `folds_for` | FunctionDef | См. реализацию | [L89](../../../../scripts/skin_local_search_train.py#L89) |
| `metrics` | FunctionDef | См. реализацию | [L104](../../../../scripts/skin_local_search_train.py#L104) |
| `center_indices` | FunctionDef | One geometric medoid per site first; round robin gives people coverage. | [L123](../../../../scripts/skin_local_search_train.py#L123) |
| `tensor` | FunctionDef | См. реализацию | [L145](../../../../scripts/skin_local_search_train.py#L145) |
| `predict` | FunctionDef | Canonical deployed float32 payload, CPU NumPy implementation. | [L149](../../../../scripts/skin_local_search_train.py#L149) |
| `SmallMLP` | ClassDef | См. реализацию | [L171](../../../../scripts/skin_local_search_train.py#L171) |
| `fit_model` | FunctionDef | См. реализацию | [L182](../../../../scripts/skin_local_search_train.py#L182) |
| `load_model` | FunctionDef | См. реализацию | [L261](../../../../scripts/skin_local_search_train.py#L261) |
| `role_summary` | FunctionDef | См. реализацию | [L266](../../../../scripts/skin_local_search_train.py#L266) |
| `lock_sources` | FunctionDef | См. реализацию | [L273](../../../../scripts/skin_local_search_train.py#L273) |
| `fit_phase` | FunctionDef | См. реализацию | [L294](../../../../scripts/skin_local_search_train.py#L294) |
| `cpu_latency` | FunctionDef | См. реализацию | [L367](../../../../scripts/skin_local_search_train.py#L367) |
| `evaluation_phase` | FunctionDef | См. реализацию | [L379](../../../../scripts/skin_local_search_train.py#L379) |
| `main` | FunctionDef | См. реализацию | [L413](../../../../scripts/skin_local_search_train.py#L413) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L149–168</summary>

```python
def predict(model, x):
    """Canonical deployed float32 payload, CPU NumPy implementation."""
    z = (np.asarray(x, dtype=np.float32) - model["x_mean"]) / model["x_std"]
    method = str(model["method"])
    if method == "mlp":
        h = z @ model["hidden_w"].T + model["hidden_b"]
        h = h / (1 + np.exp(-np.clip(h, -80, 80)))
        out = h @ model["out_w"].T + model["out_b"] + z @ model["skip_w"].T
    elif method == "ridge":
        out = np.column_stack((np.ones(len(z), dtype=np.float32), z)) @ model["beta"]
    else:
        c = model["centers"]
        # The squared-distance identity avoids N x centers x 36 temporary arrays.
        d2 = np.maximum((z * z).sum(1)[:, None] + (c * c).sum(1)[None, :] - 2 * z @ c.T, 0) / z.shape[1]
        h = np.exp(-.5 * d2 / model["widths"] ** 2)
        if method == "krr":
            out = h @ model["beta"]
        else:
            out = np.column_stack((np.ones(len(z), dtype=np.float32), z, h)) @ model["beta"]
    return out * model["y_std"] + model["y_mean"]
```

</details>

<details><summary>__init__ · L172–176</summary>

```python
def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(36, 64)
        self.out = nn.Linear(64, 3)
        self.skip = nn.Linear(36, 3, bias=False)
```

</details>

<details><summary>forward · L178–179</summary>

```python
def forward(self, x):
        return self.out(torch.nn.functional.silu(self.hidden(x))) + self.skip(x)
```

</details>
