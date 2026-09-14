# `scripts/profile_public_cc.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/profile_public_cc.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Unoptimized batch-one real input path, including CPU experts and error head.

SHA-256 исходника: `9afb78a1559465778500f2596bc2d2745f57321a7f5d42d95d3700d801460f2e`. Строк: **83**.

## Зависимости

```python
import json
import time
import zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
from luma_skin_vision.cc.benchmark import apply_risk, risk_features
from luma_skin_vision.cc.core import experts
from luma_skin_vision.cc.data import decode, sample
from luma_skin_vision.cc.model import CompactCC
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
