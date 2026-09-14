# `scripts/skin_issa_overlap.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_issa_overlap.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Post-hoc sensitivity to duplicate spectra; no refitting or split rewriting.

SHA-256 исходника: `4b8b5b1e438f4be0b73e0355e04c9769c6dc8ed6fe660c12fef5f0a7a9a1ffce`. Строк: **68**.

## Зависимости

```python
from collections import defaultdict
import hashlib
import json
import numpy as np
from skin_issa_data import OUT, PRIVATE, ROOT, sha, write_json
from skin_issa_material import load_cache, verify_lock, stats
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L10](../../../../scripts/skin_issa_overlap.py#L10) |
