# `scripts/chromaseed_affine_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_affine_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

All-policy A evidence: augmentation is not a universal quality improvement.

SHA-256 исходника: `9f66173d9a0b69334da5f16acd85dfe9a6cc617a72872acf4d14cdeaf5cda215`. Строк: **270**.

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

[Строка 17](../../../../scripts/chromaseed_affine_report.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_affine_report.py#L18)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 19](../../../../scripts/chromaseed_affine_report.py#L19)

```python
LABELS = ("Mixed · 6 people", "SLR → iPod · 16 people", "iPod → SLR · 8 people")
```

[Строка 20](../../../../scripts/chromaseed_affine_report.py#L20)

```python
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
```

[Строка 21](../../../../scripts/chromaseed_affine_report.py#L21)

```python
CONTROLS = ("g_norm_base", "g_norm_soft", "g_perceptual_base", "g_perceptual_soft", "constant")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `csv_write` | FunctionDef | См. реализацию | [L24](../../../../scripts/chromaseed_affine_report.py#L24) |
| `main` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_affine_report.py#L31) |
