# `scripts/skin_nuisance_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_nuisance_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent metric and source-boundary audit plus exact GPU checkpoint replay.

SHA-256 исходника: `e20dd9e1b8af174280023b6a0a9647000cc3edac910b106769927b0538a38b2d`. Строк: **78**.

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
from skin_nuisance_model import NuisanceColor as SpatialColor
from skin_nuisance_train import OUT,RUN,PROTOCOL,prediction,subset
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_nuisance_audit.py#L15) |
