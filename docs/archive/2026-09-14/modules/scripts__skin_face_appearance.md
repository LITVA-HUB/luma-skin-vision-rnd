# `scripts/skin_face_appearance.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_appearance.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Mask-induced apparent-color error; never an instrument skin-color benchmark.

SHA-256 исходника: `1a235576da5e1572accc35387f4f44a6400c5b2a7bb4cc3652ef38aa56733bd9`. Строк: **129**.

## Зависимости

```python
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import SkinUNet, binary_mask
from skin_lapa_prepare import file_sha
from threadpoolctl import threadpool_limits
from luma_skin_vision.color import delta_e00, srgb_to_lab
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/skin_face_appearance.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 21](../../../../scripts/skin_face_appearance.py#L21)

```python
OUT = DATA_ROOT/'facial_skin_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `appearance` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_face_appearance.py#L24) |
| `color_error` | FunctionDef | См. реализацию | [L35](../../../../scripts/skin_face_appearance.py#L35) |
| `freeze` | FunctionDef | См. реализацию | [L42](../../../../scripts/skin_face_appearance.py#L42) |
| `run` | FunctionDef | См. реализацию | [L62](../../../../scripts/skin_face_appearance.py#L62) |
