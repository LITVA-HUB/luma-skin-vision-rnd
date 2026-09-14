# `scripts/chromaseed_gaussian_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gaussian_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Describe the frozen TG experiment, including all losses and measured costs.

SHA-256 исходника: `a645c72766cffa7dc592b4d7cd16f19cbd1206a8ac40eedb7814e7c2d75a2859`. Строк: **437**.

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

[Строка 16](../../../../scripts/chromaseed_gaussian_report.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_gaussian_report.py#L17)

```python
RUN = ROOT / "experiments/runs/chromaseed_gaussian_v1"
```

[Строка 18](../../../../scripts/chromaseed_gaussian_report.py#L18)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_gaussian_v1"
```

[Строка 19](../../../../scripts/chromaseed_gaussian_report.py#L19)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-gaussian-2026-09-13.md"
```

[Строка 20](../../../../scripts/chromaseed_gaussian_report.py#L20)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 21](../../../../scripts/chromaseed_gaussian_report.py#L21)

```python
METHODS = ("adam", "tagi_diag", "tagi_full3")
```

[Строка 22](../../../../scripts/chromaseed_gaussian_report.py#L22)

```python
GROUPS = ("raw36", "mean3")
```

[Строка 23](../../../../scripts/chromaseed_gaussian_report.py#L23)

```python
EPOCHS = (1, 4, 16, 64)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `csv_write` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_gaussian_report.py#L26) |
| `same` | FunctionDef | См. реализацию | [L33](../../../../scripts/chromaseed_gaussian_report.py#L33) |
| `main` | FunctionDef | См. реализацию | [L37](../../../../scripts/chromaseed_gaussian_report.py#L37) |
