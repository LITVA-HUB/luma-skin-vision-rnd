# `scripts/chromaseed_affine_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_affine_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

A sequential standalone queries and112 complete fit repetitions after audit.

SHA-256 исходника: `2ad27edb481f16051d53e3ec92789da77eac717bebc3954ea3388f7e6a331907`. Строк: **206**.

## Зависимости

```python
from __future__ import annotations
import argparse
import gc
import os
import platform
import time
from pathlib import Path
import numpy as np
from chromaseed_affine import FAMILIES, fit_single, model_id
from chromaseed_gated_audit import model_from
from chromaseed_gated_numpy import Predictor
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_affine_runtime.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Constant` | — | [L22](../../../../scripts/chromaseed_affine_runtime.py#L22) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Constant` | ClassDef | Only the fixed native-Lab negative control, with the same input checks. | [L22](../../../../scripts/chromaseed_affine_runtime.py#L22) |
| `query_time` | FunctionDef | См. реализацию | [L42](../../../../scripts/chromaseed_affine_runtime.py#L42) |
| `timing_fit` | FunctionDef | См. реализацию | [L74](../../../../scripts/chromaseed_affine_runtime.py#L74) |
| `main` | FunctionDef | См. реализацию | [L104](../../../../scripts/chromaseed_affine_runtime.py#L104) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L25–29</summary>

```python
def __init__(self, model):
        assert set(model) == {"constant_lab"}
        value = model["constant_lab"]
        assert value.shape == (3,) and value.dtype == np.float32 and np.isfinite(value).all()
        self.value = value.astype(np.float64)
```

</details>

<details><summary>__call__ · L35–39</summary>

```python
def __call__(self, color):
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (36,) or not np.isfinite(color).all():
            raise ValueError("one finite color36 vector required")
        return self.value.copy()
```

</details>
