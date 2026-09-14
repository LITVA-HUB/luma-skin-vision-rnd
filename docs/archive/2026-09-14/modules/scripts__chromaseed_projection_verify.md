# `scripts/chromaseed_projection_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_projection_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

X integrity receipt; inherited A verifier is explicitly read-only on existing receipt.

SHA-256 исходника: `4911621239b8d5764fc9c9e06d1df73a514a83c69233936636e5ff182b053767`. Строк: **339**.

## Зависимости

```python
from __future__ import annotations
import argparse
import importlib.metadata
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote
import numpy as np
from chromaseed_gated_verify import check_map, command, read
from skin_local_search_train import CACHE_HASH, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_projection_verify.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_projection_verify.py#L18)

```python
RUN = ROOT / "experiments/runs/chromaseed_projection_v1"
```

[Строка 19](../../../../scripts/chromaseed_projection_verify.py#L19)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_projection_v1"
```

[Строка 20](../../../../scripts/chromaseed_projection_verify.py#L20)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-projection-2026-09-13.md"
```

[Строка 21](../../../../scripts/chromaseed_projection_verify.py#L21)

```python
SOURCE = "31558592fca6b0b23f22448ac61e8e86fc27da694112b2f9394160e4a5e29ecc"
```

[Строка 22](../../../../scripts/chromaseed_projection_verify.py#L22)

```python
SELECTION = "134a11bb56bfac8320ca48efe7c3f7c71e971a65147badde4f7d0500edc85e5a"
```

[Строка 23](../../../../scripts/chromaseed_projection_verify.py#L23)

```python
RESULT = "693280fab39c123e2e7c455b7b5215bf372d564e08551cbb857df9ad7097997e"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_projection_verify.py#L26) |
