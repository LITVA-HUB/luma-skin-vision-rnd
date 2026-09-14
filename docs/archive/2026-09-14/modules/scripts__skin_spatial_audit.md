# `scripts/skin_spatial_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_spatial_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent metric and source-boundary audit plus exact GPU checkpoint replay.

SHA-256 исходника: `fdef404668bcf222a06ce68a7be500c39f71b3795ea53e7b647b91a342e7d1e5`. Строк: **67**.

## Зависимости

```python
import os
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_spatial_model import SpatialColor
from skin_spatial_train import OUT,RUN,PROTOCOL,prediction,subset
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_spatial_audit.py#L15) |
