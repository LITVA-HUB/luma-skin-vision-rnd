# `scripts/chromaseed_architecture_scale_plot.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_architecture_scale_plot.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Standalone scientific figure from a provisional, source-bound inner review.

SHA-256 исходника: `63ea7a59102e1bc3cce8c6b3b1e9ce320c2d0fefd34f6fa0ab42eb0493b67fe4`. Строк: **168**.

## Зависимости

```python
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, check_map
from chromaseed_kernel_audit import js
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_architecture_scale_plot.py#L19)

```python
STYLE = {
    "patch_small": ("Участки · 17,4 тыс.", "#246B9A", "o"),
    "patch5m": ("Участки · 4,96 млн", "#D77825", "s"),
    "soft_small": ("4 уточнения · 15,2 тыс.", "#78803C", "^"),
    "soft5m": ("4 уточнения · 4,85 млн", "#A34E78", "D"),
}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_architecture_scale_plot.py#L27) |
