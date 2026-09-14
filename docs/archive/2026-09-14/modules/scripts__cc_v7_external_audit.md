# `scripts/cc_v7_external_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_external_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent scalar atan2 audit of locked real-camera benchmark reporting.

SHA-256 исходника: `6ffe1d8216e1925016ed7b262f1c6174af311449838da93dafc5bd4dd2b44b55`. Строк: **200**.

## Зависимости

```python
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from cc_v7_external_data import CACHE
from cc_v7_external_lock import BENCH, ROOT, checked_lock, load
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `angle` | FunctionDef | См. реализацию | [L17](../../../../scripts/cc_v7_external_audit.py#L17) |
| `percentile` | FunctionDef | См. реализацию | [L26](../../../../scripts/cc_v7_external_audit.py#L26) |
| `statistics` | FunctionDef | См. реализацию | [L33](../../../../scripts/cc_v7_external_audit.py#L33) |
| `run` | FunctionDef | См. реализацию | [L59](../../../../scripts/cc_v7_external_audit.py#L59) |
