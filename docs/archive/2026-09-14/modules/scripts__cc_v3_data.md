# `scripts/cc_v3_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v3_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Freeze and acquire only previously unused INTEL-TAU members, with original terms.

SHA-256 исходника: `478069011c7c8ae6e3132bd9bc53506c7496e852f89f51ee0b636f46cc4dde54`. Строк: **131**.

## Зависимости

```python
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import download_cc_v2_fresh as downloader
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../scripts/cc_v3_data.py#L13)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 14](../../../../scripts/cc_v3_data.py#L14)

```python
PROVENANCE = ROOT / "docs/data/provenance/cc_v3"
```

[Строка 15](../../../../scripts/cc_v3_data.py#L15)

```python
DATA = ROOT / "data/public/intel_tau_v3"
```

[Строка 16](../../../../scripts/cc_v3_data.py#L16)

```python
BASE = "ea490d3"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `digest` | FunctionDef | См. реализацию | [L19](../../../../scripts/cc_v3_data.py#L19) |
| `remainder` | FunctionDef | См. реализацию | [L23](../../../../scripts/cc_v3_data.py#L23) |
| `original` | FunctionDef | См. реализацию | [L36](../../../../scripts/cc_v3_data.py#L36) |
| `freeze` | FunctionDef | См. реализацию | [L44](../../../../scripts/cc_v3_data.py#L44) |
| `acquire` | FunctionDef | См. реализацию | [L102](../../../../scripts/cc_v3_data.py#L102) |
