# `scripts/chromaseed_gaussian_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gaussian_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seal TG once and verify the complete canonical evidence chain read-only later.

SHA-256 исходника: `9f71b8da52326bc5eae93886a85fe00213682a4cb9c3a6952bb1457b77260363`. Строк: **454**.

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

[Строка 19](../../../../scripts/chromaseed_gaussian_verify.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 20](../../../../scripts/chromaseed_gaussian_verify.py#L20)

```python
RUN = ROOT / "experiments/runs/chromaseed_gaussian_v1"
```

[Строка 21](../../../../scripts/chromaseed_gaussian_verify.py#L21)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_gaussian_v1"
```

[Строка 22](../../../../scripts/chromaseed_gaussian_verify.py#L22)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-gaussian-2026-09-13.md"
```

[Строка 23](../../../../scripts/chromaseed_gaussian_verify.py#L23)

```python
SOURCE = "c5f2d9f4a4a9111b4da1a5b238c0d48ecd6c851a808efb272d66e5c92cebca62"
```

[Строка 24](../../../../scripts/chromaseed_gaussian_verify.py#L24)

```python
SELECTION = "22a8c9543614ced61fb1809e95fe3ec31c12710446db0b1261080d89b47bab9b"
```

[Строка 25](../../../../scripts/chromaseed_gaussian_verify.py#L25)

```python
RESULT = "2ae1278362c022d86a31102b88db86af03cec236baec193e32274ce36c437e8f"
```

[Строка 26](../../../../scripts/chromaseed_gaussian_verify.py#L26)

```python
FG_RECEIPT = "459e6554bfba7ce22c63f7377556648632ba07406d20e1d50a42351d8bbf8908"
```

[Строка 27](../../../../scripts/chromaseed_gaussian_verify.py#L27)

```python
METHODS = ("adam", "tagi_diag", "tagi_full3")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `matched` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_gaussian_verify.py#L30) |
| `aggregate_checks` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_gaussian_verify.py#L34) |
| `main` | FunctionDef | См. реализацию | [L177](../../../../scripts/chromaseed_gaussian_verify.py#L177) |
