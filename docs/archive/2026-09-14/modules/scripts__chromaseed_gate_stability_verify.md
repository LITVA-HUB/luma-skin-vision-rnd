# `scripts/chromaseed_gate_stability_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gate_stability_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Freeze the GS verification receipt after independent numerical and source checks.

SHA-256 исходника: `5b68bbaa4e833a5dbfc3ad41f10e69d9f19288bdc3249fdd3b1a0c420bbc91c6`. Строк: **256**.

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
from chromaseed_gated_verify import check_map, command, read
from skin_local_search_train import CACHE_HASH, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_gate_stability_verify.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_gate_stability_verify.py#L17)

```python
RUNS = ROOT / "experiments/runs"
```

[Строка 18](../../../../scripts/chromaseed_gate_stability_verify.py#L18)

```python
RUN = RUNS / "chromaseed_gate_stability_v1"
```

[Строка 19](../../../../scripts/chromaseed_gate_stability_verify.py#L19)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_gate_stability_v1"
```

[Строка 20](../../../../scripts/chromaseed_gate_stability_verify.py#L20)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-stability-2026-09-13.md"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_gate_stability_verify.py#L23) |
