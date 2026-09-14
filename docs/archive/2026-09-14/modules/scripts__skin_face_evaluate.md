# `scripts/skin_face_evaluate.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_evaluate.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

First held-image evaluation only after the segmentation checkpoint is frozen.

SHA-256 исходника: `df59649c1228c42d84ddf478a4286fec7b326b28087e3bb1a142e2718871b019`. Строк: **135**.

## Зависимости

```python
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import SkinUNet, binary_mask, scores
from skin_face_train import DATA, OUT, ROOT, evaluate, setup
from skin_lapa_prepare import file_sha
from threadpoolctl import threadpool_limits
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_face_evaluate.py#L19) |
