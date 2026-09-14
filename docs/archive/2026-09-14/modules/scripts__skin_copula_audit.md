# `scripts/skin_copula_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_copula_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent metric and source-boundary audit plus exact GPU checkpoint replay.

SHA-256 исходника: `1d3a7477b985452733d5e82eff2621b5612ff49414fa0cdf1415ca3c82560578`. Строк: **67**.

## Зависимости

```python
import os
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_copula_data import load,pack
from skin_mskcc_audit import scalar_de
from skin_copula_model import DistributionColor as SpatialColor
from skin_copula_train import OUT,RUN,PROTOCOL,prediction,subset
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_copula_audit.py#L15) |
