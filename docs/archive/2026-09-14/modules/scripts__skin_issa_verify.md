# `scripts/skin_issa_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_issa_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent scalar color checks and exact-spectrum overlap diagnostic.

SHA-256 исходника: `03b7fc869e7b127a63efc4adef0e7bcd21e6ef1ae0d9f44ca7c766b4abbe5ec9`. Строк: **66**.

## Зависимости

```python
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from skin_issa_data import OUT, PRIVATE, ROOT, checked_manifest, sha, write_json
from skin_issa_material import load_cache, verify_lock
from skin_mskcc_audit import scalar_de
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_issa_verify.py#L15) |
