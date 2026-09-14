# `scripts/chromaseed_neural_shrinkage_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_shrinkage_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Generate all NS comparisons, then seal or verify them read-only.

SHA-256 исходника: `1d2c5a7b1fe5193b0a254e20a666fd9c41de06aacc308cfe3f9b728783cf2e52`. Строк: **344**.

## Зависимости

```python
from __future__ import annotations
import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_neural_shrinkage_report.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_neural_shrinkage_report.py#L17)

```python
RUN = ROOT / "experiments/runs/chromaseed_neural_shrinkage_v1"
```

[Строка 18](../../../../scripts/chromaseed_neural_shrinkage_report.py#L18)

```python
NR = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
```

[Строка 19](../../../../scripts/chromaseed_neural_shrinkage_report.py#L19)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_neural_shrinkage_v1"
```

[Строка 20](../../../../scripts/chromaseed_neural_shrinkage_report.py#L20)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-neural-shrinkage-2026-09-13.md"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `key` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_neural_shrinkage_report.py#L23) |
| `check_map` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_neural_shrinkage_report.py#L27) |
| `main` | FunctionDef | См. реализацию | [L32](../../../../scripts/chromaseed_neural_shrinkage_report.py#L32) |
