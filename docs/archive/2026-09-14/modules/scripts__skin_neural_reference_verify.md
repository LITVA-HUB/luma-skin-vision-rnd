# `scripts/skin_neural_reference_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_neural_reference_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent NumPy heads/augmented solves, scalar colors and deterministic refits.

SHA-256 исходника: `9fbb0383dff611b169de82c92583904cc89e4d6709dc2c68dac2b445e0ec4a46`. Строк: **139**.

## Зависимости

```python
import os
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from skin_neural_reference_train import OUT,RUN,PROTOCOL,bindings,prepare_core,risk_order
from skin_neural_reference import ColorAdapter,DeployedColor
from skin_local_reference_transfer import banks
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `numpy_head` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_neural_reference_verify.py#L19) |
| `independent_local` | FunctionDef | См. реализацию | [L28](../../../../scripts/skin_neural_reference_verify.py#L28) |
| `numeric_metrics` | FunctionDef | См. реализацию | [L39](../../../../scripts/skin_neural_reference_verify.py#L39) |
| `compare_metrics` | FunctionDef | См. реализацию | [L46](../../../../scripts/skin_neural_reference_verify.py#L46) |
| `main` | FunctionDef | См. реализацию | [L50](../../../../scripts/skin_neural_reference_verify.py#L50) |
