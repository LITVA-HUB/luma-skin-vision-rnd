# `scripts/chromaseed_crossfit_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_crossfit_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

C evidence tables and report; neither frozen H settings nor C outcomes are retuned.

SHA-256 исходника: `e3d691a03e2a318fa8ee906eff523f84a0872b48acee5206a666ff6ea737bf49`. Строк: **353**.

## Зависимости

```python
from __future__ import annotations
import csv
import re
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_crossfit_report.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_crossfit_report.py#L18)

```python
RUN = ROOT / "experiments/runs/chromaseed_crossfit_v1"
```

[Строка 19](../../../../scripts/chromaseed_crossfit_report.py#L19)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_crossfit_v1"
```

[Строка 20](../../../../scripts/chromaseed_crossfit_report.py#L20)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-crossfit-2026-09-13.md"
```

[Строка 21](../../../../scripts/chromaseed_crossfit_report.py#L21)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `csv_write` | FunctionDef | См. реализацию | [L24](../../../../scripts/chromaseed_crossfit_report.py#L24) |
| `main` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_crossfit_report.py#L31) |
