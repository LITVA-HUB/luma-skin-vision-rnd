# `scripts/chromaseed_refine_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_refine_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate the locked ChromaSeed-R study without selecting from outer data.

SHA-256 исходника: `697c41dd2753702618a6b3bc27d97423f255656fe5cf3d0b447d953d4cfae1cf`. Строк: **209**.

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

[Строка 17](../../../../scripts/chromaseed_refine_report.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_refine_report.py#L18)

```python
LABELS = {"stats_mlp": "Statistics MLP", "patch_mlp": "Patch MLP", "recur_soft": "Refine all 64",
          "recur_top16": "Refine top 16", "recur_dynamic": "Refine dynamic"}
```

[Строка 20](../../../../scripts/chromaseed_refine_report.py#L20)

```python
ROLES = {"mixed": "Mixed cameras, 18/6 people", "slr_to_ipod": "SLR → iPod, 8/16 people", "ipod_to_slr": "iPod → SLR, 16/8 people"}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read_json` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_refine_report.py#L23) |
| `summarize` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_refine_report.py#L27) |
| `plot` | FunctionDef | См. реализацию | [L92](../../../../scripts/chromaseed_refine_report.py#L92) |
| `report` | FunctionDef | См. реализацию | [L152](../../../../scripts/chromaseed_refine_report.py#L152) |
| `main` | FunctionDef | См. реализацию | [L200](../../../../scripts/chromaseed_refine_report.py#L200) |
