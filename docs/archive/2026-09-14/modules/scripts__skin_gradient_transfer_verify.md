# `scripts/skin_gradient_transfer_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_gradient_transfer_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent batch reduction, scalar skin color and finite-update replay.

SHA-256 исходника: `7945f84a36e283d1e4599bdbe4ac7af5f50bb2d0396145aa395e4b74ea46af7c`. Строк: **133**.

## Зависимости

```python
import os
import json,math
from pathlib import Path
import numpy as np
import torch
from skin_gradient_transfer_run import OUT,RUN,SOURCE,MODELS,bindings,setup,values
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `independent_grad` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_gradient_transfer_verify.py#L16) |
| `independent_step` | FunctionDef | См. реализацию | [L31](../../../../scripts/skin_gradient_transfer_verify.py#L31) |
| `main` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_gradient_transfer_verify.py#L36) |
