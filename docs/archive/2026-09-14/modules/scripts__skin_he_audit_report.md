# `scripts/skin_he_audit_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_he_audit_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent scalar XYZ audit and bounded real-skin calibration report.

SHA-256 исходника: `5d17bc767eaa00e110e8e57cfd637156a3a6830caaf68d19bcda7fb59d13fe48`. Строк: **110**.

## Зависимости

```python
import hashlib
import json
import math
from pathlib import Path
import joblib
import numpy as np
from cc_v7_external_audit import percentile
from skin_he_xyz import BENCH, DATA, MODELS, ROOT, inputs, read, sha, write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `scalar_stats` | FunctionDef | См. реализацию | [L33](../../../../scripts/skin_he_audit_report.py#L33) |
