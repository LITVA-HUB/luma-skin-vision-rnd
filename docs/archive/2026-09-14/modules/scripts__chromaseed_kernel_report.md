# `scripts/chromaseed_kernel_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_kernel_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-linked descriptive report of the completed frozen K experiment.

SHA-256 исходника: `34ebd0627f783ee4da611126e1c24d2b3b82bbfd92bcd8ee6bf6431a2b00712e`. Строк: **213**.

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

[Строка 16](../../../../scripts/chromaseed_kernel_report.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_kernel_report.py#L17)

```python
ROLES = {"mixed": "Смешанные камеры (6 человек)", "slr_to_ipod": "SLR → iPod (16 человек)", "ipod_to_slr": "iPod → SLR (8 человек)"}
```

[Строка 18](../../../../scripts/chromaseed_kernel_report.py#L18)

```python
EN_ROLES = {"mixed": "Mixed cameras; 6 held people", "slr_to_ipod": "SLR → iPod; 16 held people", "ipod_to_slr": "iPod → SLR; 8 held people"}
```

[Строка 19](../../../../scripts/chromaseed_kernel_report.py#L19)

```python
LABELS = {"exact": "Полное ядро", "nys_random": "Nyström random", "nys_pivot": "Nyström greedy",
          "nys_rpchol": "Nyström RPCholesky", "project_rpchol": "Проекция RPCholesky",
          "adaptive_project": "Проекция с ранним выходом", "blend_nys_rpchol_64": "Nyström64 + guided RBF",
          "blend_project_rpchol_128": "Проекция128 + guided RBF"}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_kernel_report.py#L25) |
| `aggregate` | FunctionDef | См. реализацию | [L29](../../../../scripts/chromaseed_kernel_report.py#L29) |
| `one` | FunctionDef | См. реализацию | [L95](../../../../scripts/chromaseed_kernel_report.py#L95) |
| `plot` | FunctionDef | См. реализацию | [L99](../../../../scripts/chromaseed_kernel_report.py#L99) |
| `report` | FunctionDef | См. реализацию | [L140](../../../../scripts/chromaseed_kernel_report.py#L140) |
| `main` | FunctionDef | См. реализацию | [L204](../../../../scripts/chromaseed_kernel_report.py#L204) |
