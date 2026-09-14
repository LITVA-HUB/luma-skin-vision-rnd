# `scripts/skin_correction_transfer_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_correction_transfer_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Replay source transfer, independent routing/numerics and complete sample refits.

SHA-256 исходника: `5d3e150665fce09f63e913f8605d2aa044595f563bb92d2a0b6db48a68b31a94`. Строк: **121**.

## Зависимости

```python
import os
import json
from pathlib import Path
import numpy as np
import torch
from skin_correction_transfer import OUT,RUN,bindings,source_file,active_lock
from skin_crossfit_correction import StableColorModel,features,ARMS
from skin_crossfit_correction_train import predict,fit_core,fit_head
from skin_crossfit_correction_verify import numpy_head
from skin_neural_reference_verify import numeric_metrics,compare_metrics
from skin_support_curve import SkinRepresentation
from skin_local_reference_transfer import banks
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L22](../../../../scripts/skin_correction_transfer_verify.py#L22) |
