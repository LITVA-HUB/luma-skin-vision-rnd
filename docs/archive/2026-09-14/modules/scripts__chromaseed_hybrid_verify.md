# `scripts/chromaseed_hybrid_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_hybrid_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Create one H integrity receipt; subsequent runs verify it without rewriting.

SHA-256 исходника: `5f6b481ff21e7690aa0e8ff9bdb3fbd5751ae405ef3eacd535b8eaeb32a0ec39`. Строк: **346**.

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
from chromaseed_kernel_audit import nz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 21](../../../../scripts/chromaseed_hybrid_verify.py#L21)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 22](../../../../scripts/chromaseed_hybrid_verify.py#L22)

```python
RUN = ROOT / "experiments/runs/chromaseed_hybrid_v1"
```

[Строка 23](../../../../scripts/chromaseed_hybrid_verify.py#L23)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_hybrid_v1"
```

[Строка 24](../../../../scripts/chromaseed_hybrid_verify.py#L24)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-hybrid-2026-09-13.md"
```

[Строка 25](../../../../scripts/chromaseed_hybrid_verify.py#L25)

```python
SOURCE = "42d85a41e8b63d15dda61ab6411b043c1c4eefd075088dc90d934d3b19505d84"
```

[Строка 26](../../../../scripts/chromaseed_hybrid_verify.py#L26)

```python
SELECTION = "c3a65f8be1af5e69724729c67b097957dcacac0c179811986d149daaef7daec3"
```

[Строка 27](../../../../scripts/chromaseed_hybrid_verify.py#L27)

```python
RESULT = "f5918a094f446a49ed26df84ac8251a3ca793d22b469289c683d7cd0c952876e"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_hybrid_verify.py#L30) |
