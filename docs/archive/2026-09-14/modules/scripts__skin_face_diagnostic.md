# `scripts/skin_face_diagnostic.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_diagnostic.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

CPU-only qualitative overlays for supplied photos and the public DAST example.

SHA-256 исходника: `ee4a734546c0f02c9745607d1786cb91c29f65c4c2feb0dde4a83a802f5eecb9`. Строк: **66**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import SkinUNet
from skin_lapa_prepare import file_sha
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_face_diagnostic.py#L13) |
