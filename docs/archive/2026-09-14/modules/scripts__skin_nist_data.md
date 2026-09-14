# `scripts/skin_nist_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_nist_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Parse original NIST triplicate spectra without inventing paired photographs.

SHA-256 исходника: `f30ab964d980b32af035988f6afdaf4ef8a4b4e57ae60b3a7889d3eba8474733`. Строк: **81**.

## Зависимости

```python
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
from pathlib import Path
import numpy as np
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `parse_nist` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_nist_data.py#L15) |
| `main` | FunctionDef | См. реализацию | [L52](../../../../scripts/skin_nist_data.py#L52) |
