# `scripts/skin_face_baseline.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_baseline.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen UCI pixel model on a pre-hashed LaPa validation subset; CPU only.

SHA-256 исходника: `8d227b43db80eb9e008d2dcd30decb0eca0515e78efecb79efb8338db014efbc`. Строк: **56**.

## Зависимости

```python
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import binary_mask, scores
from skin_lapa_prepare import file_sha
from threadpoolctl import threadpool_limits
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_face_baseline.py#L15) |
