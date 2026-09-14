# `scripts/audit_cc_v2_metrics.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/audit_cc_v2_metrics.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Reproduce the frozen v2 audit, writing a new receipt without changing evidence.

SHA-256 исходника: `cfd97ae7dec8b43aa2e99a39f3bc28cb3a58ec5b6f80b6b747610b4368a358d3`. Строк: **799**.

## Зависимости

```python
import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 23](../../../../scripts/audit_cc_v2_metrics.py#L23)

```python
ROOT = args.repo_root.resolve()
```

[Строка 24](../../../../scripts/audit_cc_v2_metrics.py#L24)

```python
BASE = ROOT / "docs/benchmarks/cc_v2"
```

[Строка 25](../../../../scripts/audit_cc_v2_metrics.py#L25)

```python
RUNS = ROOT / "experiments/runs"
```

[Строка 55](../../../../scripts/audit_cc_v2_metrics.py#L55)

```python
TOL = 1e-7
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `local_path` | FunctionDef | См. реализацию | [L44](../../../../scripts/audit_cc_v2_metrics.py#L44) |
| `load` | FunctionDef | См. реализацию | [L58](../../../../scripts/audit_cc_v2_metrics.py#L58) |
| `sha` | FunctionDef | См. реализацию | [L62](../../../../scripts/audit_cc_v2_metrics.py#L62) |
| `ok` | FunctionDef | См. реализацию | [L71](../../../../scripts/audit_cc_v2_metrics.py#L71) |
| `binding` | FunctionDef | См. реализацию | [L77](../../../../scripts/audit_cc_v2_metrics.py#L77) |
| `diff` | FunctionDef | См. реализацию | [L82](../../../../scripts/audit_cc_v2_metrics.py#L82) |
| `angle` | FunctionDef | См. реализацию | [L109](../../../../scripts/audit_cc_v2_metrics.py#L109) |
| `numeric_delta` | FunctionDef | См. реализацию | [L117](../../../../scripts/audit_cc_v2_metrics.py#L117) |
| `errors` | FunctionDef | См. реализацию | [L130](../../../../scripts/audit_cc_v2_metrics.py#L130) |
| `summary` | FunctionDef | См. реализацию | [L135](../../../../scripts/audit_cc_v2_metrics.py#L135) |
| `order_for` | FunctionDef | См. реализацию | [L154](../../../../scripts/audit_cc_v2_metrics.py#L154) |
