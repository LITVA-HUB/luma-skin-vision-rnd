# `scripts/skin_mskcc_acquire_images.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_acquire_images.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Acquire only instrument-paired original images using licensed ISIC API records.

SHA-256 исходника: `d77d5d2cad5ec8387f1f9f36779280e25a7325947c5ce48ec467e2d1d49adf82`. Строк: **103**.

## Зависимости

```python
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from skin_mskcc_data import RAW, PROV, manifest, sha, MANIFEST
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fetch` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_mskcc_acquire_images.py#L13) |
| `one` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_mskcc_acquire_images.py#L27) |
| `main` | FunctionDef | См. реализацию | [L70](../../../../scripts/skin_mskcc_acquire_images.py#L70) |
