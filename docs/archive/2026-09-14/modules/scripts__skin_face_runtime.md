# `scripts/skin_face_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Measured CPU inference latency after training/testing; includes a JPEG-to-mask path.

SHA-256 исходника: `d728b692ecbb78603c4658a4a4a0bf02e4878c208e870fad067d21e512301757`. Строк: **110**.

## Зависимости

```python
import io
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import SkinUNet
from skin_lapa_prepare import file_sha
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `describe` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_face_runtime.py#L16) |
| `tensor_from_rgb` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_face_runtime.py#L24) |
| `main` | FunctionDef | См. реализацию | [L29](../../../../scripts/skin_face_runtime.py#L29) |
