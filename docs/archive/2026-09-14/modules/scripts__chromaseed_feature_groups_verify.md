# `scripts/chromaseed_feature_groups_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_feature_groups_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seal FG once; subsequent verification and the C/H/X/A chain are read-only.

SHA-256 исходника: `dba718cf1aea6088e6a619ce6fe6957ac0f994cd229c34f5b473b752fd219a26`. Строк: **458**.

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

[Строка 19](../../../../scripts/chromaseed_feature_groups_verify.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 20](../../../../scripts/chromaseed_feature_groups_verify.py#L20)

```python
RUN = ROOT / "experiments/runs/chromaseed_feature_groups_v1"
```

[Строка 21](../../../../scripts/chromaseed_feature_groups_verify.py#L21)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_feature_groups_v1"
```

[Строка 22](../../../../scripts/chromaseed_feature_groups_verify.py#L22)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-feature-groups-2026-09-13.md"
```

[Строка 23](../../../../scripts/chromaseed_feature_groups_verify.py#L23)

```python
SOURCE = "f6a8ac340650ae5d62004f59916ba720b702e6dc38f1fedf6928f5797b53f4cf"
```

[Строка 24](../../../../scripts/chromaseed_feature_groups_verify.py#L24)

```python
SELECTION = "595f9e44cf31f0d354147d09e1293600ee2cde602727578d3db0525b500ff031"
```

[Строка 25](../../../../scripts/chromaseed_feature_groups_verify.py#L25)

```python
RESULT = "7c0007b1e35e675901227340483cf156e0c80703d0d44076c520531235934250"
```

[Строка 26](../../../../scripts/chromaseed_feature_groups_verify.py#L26)

```python
C_RECEIPT = "80c4129565931e204d663d4e833d8513e7d70111b47c965b6ffe63f2eb2c18dd"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `aggregate_checks` | FunctionDef | См. реализацию | [L29](../../../../scripts/chromaseed_feature_groups_verify.py#L29) |
| `main` | FunctionDef | См. реализацию | [L164](../../../../scripts/chromaseed_feature_groups_verify.py#L164) |
