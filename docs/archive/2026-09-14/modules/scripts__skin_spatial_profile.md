# `scripts/skin_spatial_profile.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_spatial_profile.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Prepared-token GPU model latency only; no claim of complete photo latency.

SHA-256 исходника: `213ef1bcb7e0db49fc81a7335fd6a8a7b55556267ec2fe950400db56d293139a`. Строк: **42**.

## Зависимости

```python
import os
import json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_spatial_model import SpatialColor,ARMS
from skin_spatial_train import OUT,RUN
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_spatial_profile.py#L14) |
