# `docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Audit original face masks and create an unsplit auxiliary pool; no training.

SHA-256 исходника: `0d1431acaf8356878bdea6962f0c0ab9a20c9a4e137734347996b40def3f1c15`. Строк: **189**.

## Зависимости

```python
import collections
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py#L14)

```python
ROOT = Path(__file__).resolve().parent
```

[Строка 15](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py#L15)

```python
ARCHIVE = ROOT / 'CelebAMask-HQ.zip'
```

[Строка 16](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py#L16)

```python
OUT = ROOT / 'prepared_192_v1'
```

[Строка 17](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py#L17)

```python
LABELS = {'skin', 'nose', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear',
          'r_ear', 'ear_r', 'u_lip', 'l_lip', 'mouth', 'hair', 'hat', 'neck', 'neck_l', 'cloth'}
```

[Строка 19](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py#L19)

```python
POSITIVE = {'skin', 'nose'}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha` | FunctionDef | См. реализацию | [L22](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py#L22) |
| `save_json` | FunctionDef | См. реализацию | [L27](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py#L27) |
| `decode_mask` | FunctionDef | См. реализацию | [L33](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py#L33) |
| `main` | FunctionDef | См. реализацию | [L48](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_pool.py#L48) |
