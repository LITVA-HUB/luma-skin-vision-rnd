# `scripts/chromaseed_affine_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_affine_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Verify A and inherited evidence; create the final receipt once, then read-only.

SHA-256 исходника: `926e18e6ec4ffa8e2207b0bc33ec8bce0661dfd49a94121ffe5873577984b66d`. Строк: **376**.

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

[Строка 17](../../../../scripts/chromaseed_affine_verify.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_affine_verify.py#L18)

```python
RUNS = ROOT / "experiments/runs"
```

[Строка 19](../../../../scripts/chromaseed_affine_verify.py#L19)

```python
RUN = RUNS / "chromaseed_affine_v1"
```

[Строка 20](../../../../scripts/chromaseed_affine_verify.py#L20)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_affine_v1"
```

[Строка 21](../../../../scripts/chromaseed_affine_verify.py#L21)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-affine-2026-09-13.md"
```

[Строка 22](../../../../scripts/chromaseed_affine_verify.py#L22)

```python
SOURCE = "dfa348cfa6663f58de56d7a09105e4f1c81f77712297b85c9b8039eb1c5cef2a"
```

[Строка 23](../../../../scripts/chromaseed_affine_verify.py#L23)

```python
SELECTION = "e2e290c7b798294e76b4cab1d142ff5e10844bece9d265adb20b5432d1e39977"
```

[Строка 24](../../../../scripts/chromaseed_affine_verify.py#L24)

```python
RESULT = "283e5863c7e0df50cb27883b9c4151f88272788789c6f8a5486828949c9cca58"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_affine_verify.py#L27) |
