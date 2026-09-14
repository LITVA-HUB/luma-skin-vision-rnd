# `scripts/chromaseed_neural_prefix_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_prefix_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seal NP measurements, reproducible tables and report; thereafter read-only.

SHA-256 исходника: `b87f3b443c83fb91688a7c3f44745a24b9aaff0da205926aada7272ee7da4822`. Строк: **426**.

## Зависимости

```python
from __future__ import annotations
import csv
import json
import re
import subprocess
import sys
import numpy as np
from chromaseed_kernel_audit import js
from chromaseed_neural_prefix_run import ND, OUT, ROOT, RUN, check_map
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_neural_prefix_report.py#L16)

```python
CARD = ROOT / "docs/architecture/chromaseed_neural_prefix_model_card.md"
```

[Строка 17](../../../../scripts/chromaseed_neural_prefix_report.py#L17)

```python
NEXT = ROOT / "docs/research/chromaseed_neural_prefix_next_decision.md"
```

[Строка 18](../../../../scripts/chromaseed_neural_prefix_report.py#L18)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-neural-prefix-2026-09-13.md"
```

[Строка 19](../../../../scripts/chromaseed_neural_prefix_report.py#L19)

```python
FAMILIES = ("plain", "local2", "local4", "blind4", "e2e4")
```

[Строка 20](../../../../scripts/chromaseed_neural_prefix_report.py#L20)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `csv_out` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_neural_prefix_report.py#L23) |
| `main` | FunctionDef | См. реализацию | [L36](../../../../scripts/chromaseed_neural_prefix_report.py#L36) |
