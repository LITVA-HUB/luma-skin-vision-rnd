# `scripts/skin_loss_field_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_loss_field_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Replay image fields, source selections and independent scalar skin metrics.

SHA-256 исходника: `f0229713ea5925e4154244f8dc9a9822d6e55902b385bd119bd0c963c6de450c`. Строк: **82**.

## Зависимости

```python
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_loss_field_train import OUT,RUN,PRIOR,PROTOCOL,prediction,gpu_palette
from skin_loss_field import LossFieldImage,ARMS
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_loss_field_audit.py#L15) |
