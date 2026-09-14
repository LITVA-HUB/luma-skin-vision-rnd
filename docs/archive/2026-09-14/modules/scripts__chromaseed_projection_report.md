# `scripts/chromaseed_projection_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_projection_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

X evidence: compact mixed-role gain with adverse camera-transfer results.

SHA-256 исходника: `a15e420eba53979208943c1e61aa166672f178678fd907d2de38864ec09c6326`. Строк: **361**.

## Зависимости

```python
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_projection_report.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_projection_report.py#L18)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 19](../../../../scripts/chromaseed_projection_report.py#L19)

```python
LABELS = ("Mixed · 6 people", "SLR → iPod · 16 people", "iPod → SLR · 8 people")
```

[Строка 20](../../../../scripts/chromaseed_projection_report.py#L20)

```python
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
```

[Строка 21](../../../../scripts/chromaseed_projection_report.py#L21)

```python
CONTROLS = (
    "g_norm_soft",
    "g_perceptual_soft",
    "a_norm_joint_guarded",
    "a_perceptual_joint_guarded",
    "constant",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `csv_write` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_projection_report.py#L30) |
| `main` | FunctionDef | См. реализацию | [L37](../../../../scripts/chromaseed_projection_report.py#L37) |
