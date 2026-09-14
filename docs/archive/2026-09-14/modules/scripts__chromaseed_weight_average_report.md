# `scripts/chromaseed_weight_average_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_weight_average_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seal WA selected means, matched controls, timing and preserved adverse evidence.

SHA-256 исходника: `44587fa59f107a4c797262671caa186fc20d99589cffbc66dee22d4b680bdc5b`. Строк: **405**.

## Зависимости

```python
from __future__ import annotations
import re
import subprocess
import sys
import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_report import csv_out
from chromaseed_long_training_run import NP
from chromaseed_refine_audit import error_summary
from chromaseed_weight_average_run import OUT, ROOT, RUN, check_map, load_data
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_weight_average_report.py#L17)

```python
CARD = ROOT / "docs/architecture/chromaseed_weight_average_model_card.md"
```

[Строка 18](../../../../scripts/chromaseed_weight_average_report.py#L18)

```python
NEXT = ROOT / "docs/research/chromaseed_weight_average_next_decision.md"
```

[Строка 19](../../../../scripts/chromaseed_weight_average_report.py#L19)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-weight-average-2026-09-13.md"
```

[Строка 20](../../../../scripts/chromaseed_weight_average_report.py#L20)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 21](../../../../scripts/chromaseed_weight_average_report.py#L21)

```python
SCRIPTS = (
    "chromaseed_weight_average",
    "chromaseed_weight_average_run",
    "chromaseed_weight_average_audit",
    "chromaseed_weight_average_runtime",
    "chromaseed_weight_average_report",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `paired` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_weight_average_report.py#L30) |
| `main` | FunctionDef | См. реализацию | [L78](../../../../scripts/chromaseed_weight_average_report.py#L78) |
