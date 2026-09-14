# `scripts/cc_v3_frame_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v3_frame_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-only conditioning audit for the proposed full color frame; no target evaluation.

SHA-256 исходника: `dc41d9de5f14a06d444bd3cceb4ea3cd3c1fe54cf0499221008505c0760f8a82`. Строк: **81**.

## Зависимости

```python
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from cc_v2_statistics import read_npz_rows
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/cc_v3_frame_audit.py#L15) |
