# `scripts/skin_sampling_transfer_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_sampling_transfer_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Known/unseen source split, native-color sampling and final-state audit.

SHA-256 исходника: `79c4d53e8dab616a12279cfca3aed2029bb362562d807a5cb41255eca528d4f4`. Строк: **99**.

## Зависимости

```python
import os
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from skin_sampling_transfer_train import OUT,RUN,bindings,predict
from skin_color_sampling_verify import independent_probability,indices_for,refit
from skin_support_curve_verify import check_summary
from skin_support_curve import SkinRepresentation
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_sampling_transfer_verify.py#L19) |
