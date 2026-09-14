# `docs/archive/2026-09-14/evidence/data_growth_2026_09_14/lapa/prepare_source_attempt1.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/lapa/prepare_source_attempt1.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Validate LaPa originals and prepare train/validation only; never decode test images.

SHA-256 исходника: `b5b3d65b1f6a572b22ddc1f4e6e384be0e00ab356bbcf578caf0febb0619f435`. Строк: **179**.

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
| `member_target` | FunctionDef | См. реализацию | [L16](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/lapa/prepare_source_attempt1.py#L16) |
| `select_rows` | FunctionDef | См. реализацию | [L27](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/lapa/prepare_source_attempt1.py#L27) |
| `file_sha` | FunctionDef | См. реализацию | [L56](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/lapa/prepare_source_attempt1.py#L56) |
| `extract` | FunctionDef | См. реализацию | [L61](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/lapa/prepare_source_attempt1.py#L61) |
| `prepare` | FunctionDef | См. реализацию | [L117](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/lapa/prepare_source_attempt1.py#L117) |
| `main` | FunctionDef | См. реализацию | [L163](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/lapa/prepare_source_attempt1.py#L163) |
