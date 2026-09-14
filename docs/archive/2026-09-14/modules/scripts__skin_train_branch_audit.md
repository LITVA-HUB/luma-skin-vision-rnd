# `scripts/skin_train_branch_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_train_branch_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent metric and source-boundary audit plus exact GPU checkpoint replay.

SHA-256 исходника: `c66855bff8f4df773ea5d87757dd6ef3ef5f995a7e12220089cf5dca68d89cba`. Строк: **74**.

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
from skin_train_branch_model import TrainingBranchColor as SpatialColor
from skin_train_branch_train import OUT,RUN,PROTOCOL,prediction,subset
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_train_branch_audit.py#L15) |
