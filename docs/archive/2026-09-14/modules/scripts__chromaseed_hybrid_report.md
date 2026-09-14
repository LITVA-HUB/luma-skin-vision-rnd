# `scripts/chromaseed_hybrid_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_hybrid_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-bound H results, all policies/doses, plots, model card and next decision.

SHA-256 исходника: `65b46d2a4dd5c34ab527c251d1d1ea03b5fad6065ba4e93a6074f892c4c11dde`. Строк: **321**.

## Зависимости

```python
from __future__ import annotations
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

[Строка 16](../../../../scripts/chromaseed_hybrid_report.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_hybrid_report.py#L17)

```python
RUN = ROOT / "experiments/runs/chromaseed_hybrid_v1"
```

[Строка 18](../../../../scripts/chromaseed_hybrid_report.py#L18)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_hybrid_v1"
```

[Строка 19](../../../../scripts/chromaseed_hybrid_report.py#L19)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-hybrid-2026-09-13.md"
```

[Строка 20](../../../../scripts/chromaseed_hybrid_report.py#L20)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 21](../../../../scripts/chromaseed_hybrid_report.py#L21)

```python
LABELS = dict(
    raw="Исходная",
    projected="Сжатая, общие опоры",
    blend="Смешивание",
    uniform="Поправка",
    support="Взвешенная поправка",
    x_fixed="Сжатая X",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `csv_write` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_hybrid_report.py#L31) |
| `main` | FunctionDef | См. реализацию | [L38](../../../../scripts/chromaseed_hybrid_report.py#L38) |
