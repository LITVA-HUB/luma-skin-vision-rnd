# `scripts/chromaseed_fast_kernel_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_fast_kernel_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Report KF search failures alongside the exact KE implementation gain.

SHA-256 исходника: `c03dc59f20f992ff545a76b4f1cf29706a737562d85bf33b0f6d2e67488f6799`. Строк: **198**.

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

[Строка 16](../../../../scripts/chromaseed_fast_kernel_report.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_fast_kernel_report.py#L17)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 18](../../../../scripts/chromaseed_fast_kernel_report.py#L18)

```python
ARMS = ("dense_exact", "column_exact", "column_pairs1024", "column_pairs4096", "column_pairs16384")
```

[Строка 19](../../../../scripts/chromaseed_fast_kernel_report.py#L19)

```python
NAMES = {"dense_exact": "Полное ядро + точная медиана", "column_exact": "Столбцы + точная медиана",
         "column_pairs1024": "Столбцы + 1024 пары", "column_pairs4096": "Столбцы + 4096 пар", "column_pairs16384": "Столбцы + 16384 пары",
         "condensed_exact": "Столбцы + точная медиана SciPy"}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L24](../../../../scripts/chromaseed_fast_kernel_report.py#L24) |
| `summarize` | FunctionDef | См. реализацию | [L28](../../../../scripts/chromaseed_fast_kernel_report.py#L28) |
| `row_for` | FunctionDef | См. реализацию | [L84](../../../../scripts/chromaseed_fast_kernel_report.py#L84) |
| `plot` | FunctionDef | См. реализацию | [L88](../../../../scripts/chromaseed_fast_kernel_report.py#L88) |
| `report` | FunctionDef | См. реализацию | [L133](../../../../scripts/chromaseed_fast_kernel_report.py#L133) |
| `main` | FunctionDef | См. реализацию | [L189](../../../../scripts/chromaseed_fast_kernel_report.py#L189) |
