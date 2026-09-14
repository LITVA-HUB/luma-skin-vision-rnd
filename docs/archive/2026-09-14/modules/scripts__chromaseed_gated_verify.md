# `scripts/chromaseed_gated_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gated_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Final G integrity receipt; no fitting, evaluation, or changes to frozen primary sources.

SHA-256 исходника: `34dcb70503fce39efcaa4a63c257d51d12817cc7027885405621b807bbc2d817`. Строк: **343**.

## Зависимости

```python
from __future__ import annotations
import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote
from skin_local_search_train import CACHE_HASH, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_gated_verify.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_gated_verify.py#L18)

```python
RUNS = ROOT / "experiments/runs"
```

[Строка 19](../../../../scripts/chromaseed_gated_verify.py#L19)

```python
RUN = RUNS / "chromaseed_gated_v1"
```

[Строка 20](../../../../scripts/chromaseed_gated_verify.py#L20)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_gated_v1"
```

[Строка 21](../../../../scripts/chromaseed_gated_verify.py#L21)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-gated-2026-09-13.md"
```

[Строка 22](../../../../scripts/chromaseed_gated_verify.py#L22)

```python
G_LOCK = "7af66943a24be2f3d41c0f8dd5ad1261ad0126471ea0c9f97d48e9069608dd2f"
```

[Строка 23](../../../../scripts/chromaseed_gated_verify.py#L23)

```python
G_SELECTION = "da046187e9fa2da00c9605d1e709b28452b0cb0128349ae92b35c83cfbb44418"
```

[Строка 24](../../../../scripts/chromaseed_gated_verify.py#L24)

```python
G_RESULTS = "0f1da381da0ff5c882aaf29c6091fc56c179a264cd60f8c1ae2838b256a18cf5"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_gated_verify.py#L27) |
| `check_map` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_gated_verify.py#L31) |
| `command` | FunctionDef | См. реализацию | [L37](../../../../scripts/chromaseed_gated_verify.py#L37) |
| `main` | FunctionDef | См. реализацию | [L42](../../../../scripts/chromaseed_gated_verify.py#L42) |
