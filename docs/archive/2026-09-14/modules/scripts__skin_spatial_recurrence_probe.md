# `scripts/skin_spatial_recurrence_probe.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_spatial_recurrence_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Post-screen inference-step ablation; never replaces the predeclared scores.

SHA-256 исходника: `c79ca2b860c38eafdac4fcf056675b714423c1e36f203f6fbc55f2fc6a997e94`. Строк: **52**.

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
| `main` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_spatial_recurrence_probe.py#L16) |
