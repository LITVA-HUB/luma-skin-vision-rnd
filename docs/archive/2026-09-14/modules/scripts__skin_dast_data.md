# `scripts/skin_dast_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_dast_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Preserve the public DAST sample and its native, site-specific measurements.

SHA-256 исходника: `b9cdd9995b032ae4b921f1b369aa57dae1776fc75b929bebe4fa9dbd1f8a3f4c`. Строк: **134**.

## Зависимости

```python
from __future__ import annotations
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from PIL import Image
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `parse_cmf` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_dast_data.py#L17) |
| `ingest_dast` | FunctionDef | См. реализацию | [L44](../../../../scripts/skin_dast_data.py#L44) |
| `main` | FunctionDef | См. реализацию | [L123](../../../../scripts/skin_dast_data.py#L123) |
