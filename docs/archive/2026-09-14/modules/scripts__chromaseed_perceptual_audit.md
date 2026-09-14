# `scripts/chromaseed_perceptual_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_perceptual_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent analytic geometry, augmented-SVD and selection audit for P.

SHA-256 исходника: `e1dddfa2516c9232393635ff88761d1a35b2160992ce3b58c4485be613f777a6`. Строк: **240**.

## Зависимости

```python
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
from chromaseed_kernel_audit import direct_kernel, independent_predict, js, norm, nz, unpack
from chromaseed_perceptual_reference import analytic_tensor, augmented_svd
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_perceptual_audit.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_perceptual_audit.py#L17)

```python
FAMILIES = ("norm_mse", "constant_de2", "local_de2", "local_irls", "midpoint_irls")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `name_for` | FunctionDef | См. реализацию | [L21](../../../../scripts/chromaseed_perceptual_audit.py#L21) |
| `balanced` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_perceptual_audit.py#L26) |
| `reference_readouts` | FunctionDef | См. реализацию | [L37](../../../../scripts/chromaseed_perceptual_audit.py#L37) |
| `audit` | FunctionDef | См. реализацию | [L93](../../../../scripts/chromaseed_perceptual_audit.py#L93) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>loss · L62–66</summary>

```python
def loss(beta):
        residual = (z @ beta - target) * std
        distance2 = (np.einsum("ni,nij,nj->n", residual, raw, residual) if family == "local_irls"
                     else delta_e00(y + residual, y) ** 2)
        return float(weights @ (4 * (np.sqrt(np.maximum(distance2, 0) + 4) - 2)) / scale + alpha * np.sum(beta ** 2))
```

</details>
