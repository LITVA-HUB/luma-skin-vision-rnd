# `src/luma_skin_vision/cc/data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/cc/data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Publisher split/GT + local reproducible capture-day partitions; never sRGB-decode.

SHA-256 исходника: `1f3836542cc57df34c1bcf3b1a31654ffcd5b4a9dd869e571f074bf3425176f6`. Строк: **138**.

## Зависимости

```python
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
import cv2
import numpy as np
from .core import experts
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `partition` | FunctionDef | См. реализацию | [L16](../../../../src/luma_skin_vision/cc/data.py#L16) |
| `decode` | FunctionDef | См. реализацию | [L21](../../../../src/luma_skin_vision/cc/data.py#L21) |
| `sample` | FunctionDef | См. реализацию | [L30](../../../../src/luma_skin_vision/cc/data.py#L30) |
| `prepare_cube` | FunctionDef | См. реализацию | [L39](../../../../src/luma_skin_vision/cc/data.py#L39) |
| `prepare_sony` | FunctionDef | См. реализацию | [L102](../../../../src/luma_skin_vision/cc/data.py#L102) |
