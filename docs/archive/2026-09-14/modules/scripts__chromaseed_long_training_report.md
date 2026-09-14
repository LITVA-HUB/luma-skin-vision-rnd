# `scripts/chromaseed_long_training_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_long_training_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seal LT curves, honest selected outcomes, model card and reconstruction receipts.

SHA-256 исходника: `447abc27c0b49e3e9b6f34586ff375901226959797b094b14f5c79410018f83e`. Строк: **449**.

## Зависимости

```python
from __future__ import annotations
import csv
import json
import re
import subprocess
import sys
import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import NP, OUT, ROOT, RUN, check_map, load_data
from chromaseed_refine_audit import error_summary
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_long_training_report.py#L17)

```python
CARD = ROOT / "docs/architecture/chromaseed_long_training_model_card.md"
```

[Строка 18](../../../../scripts/chromaseed_long_training_report.py#L18)

```python
NEXT = ROOT / "docs/research/chromaseed_long_training_next_decision.md"
```

[Строка 19](../../../../scripts/chromaseed_long_training_report.py#L19)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-long-training-2026-09-13.md"
```

[Строка 20](../../../../scripts/chromaseed_long_training_report.py#L20)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 21](../../../../scripts/chromaseed_long_training_report.py#L21)

```python
SCRIPTS = (
    "chromaseed_long_training_fit",
    "chromaseed_long_training_run",
    "chromaseed_long_training_audit",
    "chromaseed_long_training_runtime",
    "chromaseed_long_training_report",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `csv_out` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_long_training_report.py#L30) |
| `pairs` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_long_training_report.py#L43) |
| `main` | FunctionDef | См. реализацию | [L87](../../../../scripts/chromaseed_long_training_report.py#L87) |
