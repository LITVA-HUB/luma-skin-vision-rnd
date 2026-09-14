# `scripts/chromaseed_gated_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gated_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Report all frozen G families, controls and actual standalone runtime measurements.

SHA-256 исходника: `59b053fe4f2284be51a9d8362244fdde96bd3d1a8e4acc5ca341af03c31ad79f`. Строк: **310**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_gated_report.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_gated_report.py#L18)

```python
ROLES = {"mixed": "Mixed", "slr_to_ipod": "SLR → iPod", "ipod_to_slr": "iPod → SLR"}
```

[Строка 19](../../../../scripts/chromaseed_gated_report.py#L19)

```python
NAMES = {
    "norm_base": "MSE base",
    "norm_uniform": "MSE uniform",
    "norm_soft": "MSE soft gate",
    "norm_hard": "MSE hard gate",
    "perceptual_base": "Perceptual base",
    "perceptual_uniform": "Perceptual uniform",
    "perceptual_soft": "Perceptual soft gate",
    "perceptual_hard": "Perceptual hard gate",
}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_gated_report.py#L31) |
| `aggregate` | FunctionDef | См. реализацию | [L35](../../../../scripts/chromaseed_gated_report.py#L35) |
| `draw` | FunctionDef | См. реализацию | [L83](../../../../scripts/chromaseed_gated_report.py#L83) |
| `main` | FunctionDef | См. реализацию | [L149](../../../../scripts/chromaseed_gated_report.py#L149) |
