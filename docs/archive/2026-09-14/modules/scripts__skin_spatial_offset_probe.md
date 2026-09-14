# `scripts/skin_spatial_offset_probe.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_spatial_offset_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Opponent control: is graph removal mostly a source-derived constant Lab shift?

SHA-256 исходника: `18dced1bba98f45f446e507229253663310ac2dc400e792195dfbb14e126bbd5`. Строк: **61**.

## Зависимости

```python
import os
import json
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_spatial_model import SpatialColor
from skin_spatial_train import OUT,RUN,prediction,subset
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_spatial_offset_probe.py#L16) |
