# `scripts/chromaseed_neural_geometry_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_geometry_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seal or read-only verify the inner-only geometry evidence.

SHA-256 исходника: `c786db45b7c47666e10c954e2e59691442f3b8a27d86d9a10b60709ddcb0051e`. Строк: **139**.

## Зависимости

```python
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_neural_geometry_report.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 15](../../../../scripts/chromaseed_neural_geometry_report.py#L15)

```python
RUN = ROOT / "experiments/runs/chromaseed_neural_geometry_v1"
```

[Строка 16](../../../../scripts/chromaseed_neural_geometry_report.py#L16)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_neural_geometry_v1"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L19](../../../../scripts/chromaseed_neural_geometry_report.py#L19) |
