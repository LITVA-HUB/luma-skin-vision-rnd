# `scripts/chromaseed_weak_ridge_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_weak_ridge_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Expanded-grid report: separate search effects from correction effects.

SHA-256 исходника: `78da7b3896ab4a2b05d0ff9e81d30c7e63c84c8191693a417d18b15bb4bbcec8`. Строк: **151**.

## Зависимости

```python
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_weak_ridge_report.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_weak_ridge_report.py#L17)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 18](../../../../scripts/chromaseed_weak_ridge_report.py#L18)

```python
FAMILIES = ("norm_mse", "constant_de2", "local_de2", "local_irls", "midpoint_irls")
```

[Строка 19](../../../../scripts/chromaseed_weak_ridge_report.py#L19)

```python
SHORT = ("Norm", "Global", "Local", "IRLS", "Midpoint")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `report` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_weak_ridge_report.py#L22) |
