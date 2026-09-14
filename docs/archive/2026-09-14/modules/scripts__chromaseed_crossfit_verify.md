# `scripts/chromaseed_crossfit_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_crossfit_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seal C once and verify existing C/H/X/A receipts without rewriting them.

SHA-256 исходника: `c52e46d49fe46b18d373705f3fbc0abbb3ad7837e356414fbdaa3b3ecd538bba`. Строк: **343**.

## Зависимости

```python
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote
import numpy as np
from chromaseed_gate_stability_audit import close
from chromaseed_gated_verify import check_map, command, read
from skin_local_search_train import CACHE_HASH, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_crossfit_verify.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 20](../../../../scripts/chromaseed_crossfit_verify.py#L20)

```python
RUN = ROOT / "experiments/runs/chromaseed_crossfit_v1"
```

[Строка 21](../../../../scripts/chromaseed_crossfit_verify.py#L21)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_crossfit_v1"
```

[Строка 22](../../../../scripts/chromaseed_crossfit_verify.py#L22)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-crossfit-2026-09-13.md"
```

[Строка 23](../../../../scripts/chromaseed_crossfit_verify.py#L23)

```python
SOURCE = "bf6b8d289bab901007d925588a90ab8337fe231291c87ce96ecb01ae0b2b808e"
```

[Строка 24](../../../../scripts/chromaseed_crossfit_verify.py#L24)

```python
SETTINGS = "6e8fe0c8e4a2035ebc271c5164c2bde09fb46fe36d1300b3c6026946e06e1041"
```

[Строка 25](../../../../scripts/chromaseed_crossfit_verify.py#L25)

```python
RESULT = "5ce1a44e759334d8f76222870b22873584e10484deb29f295b9cc563c87b391b"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L28](../../../../scripts/chromaseed_crossfit_verify.py#L28) |
