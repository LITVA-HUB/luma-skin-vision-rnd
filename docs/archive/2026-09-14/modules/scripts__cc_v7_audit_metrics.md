# `scripts/cc_v7_audit_metrics.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_audit_metrics.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent scalar/ranking audit of recorded V7 source and virtual diagnostics.

SHA-256 исходника: `76cca82f3740b707ecba0d7fb3a813759887787e377c8e03381c1785de0e42f6`. Строк: **70**.

## Зависимости

```python
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cc_v7_verify import score, verify_manifest
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../scripts/cc_v7_audit_metrics.py#L13)

```python
ROOT=Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `run` | FunctionDef | См. реализацию | [L16](../../../../scripts/cc_v7_audit_metrics.py#L16) |
