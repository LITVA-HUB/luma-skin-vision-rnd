# `scripts/skin_lapa_prepare.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_lapa_prepare.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Validate LaPa originals and prepare train/validation only; never decode test images.

SHA-256 исходника: `4f9b715851bafce7718efb74e05db61caccc59515c00ac9fe578e2e2157d2ca1`. Строк: **184**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `member_target` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_lapa_prepare.py#L16) |
| `select_rows` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_lapa_prepare.py#L27) |
| `file_sha` | FunctionDef | См. реализацию | [L56](../../../../scripts/skin_lapa_prepare.py#L56) |
| `extract` | FunctionDef | См. реализацию | [L61](../../../../scripts/skin_lapa_prepare.py#L61) |
| `prepare` | FunctionDef | См. реализацию | [L120](../../../../scripts/skin_lapa_prepare.py#L120) |
| `main` | FunctionDef | См. реализацию | [L168](../../../../scripts/skin_lapa_prepare.py#L168) |
