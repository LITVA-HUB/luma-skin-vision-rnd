# `scripts/skin_crossfit_correction_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_crossfit_correction_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Encoder-role/scaling audit, numeric replay and complete fit reproducibility.

SHA-256 исходника: `c93dc6b5088319989a5187da48c64ecaaa77bb6215a91fbb0791e2dfd5d4b870`. Строк: **116**.

## Зависимости

```python
import os
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from skin_crossfit_correction_train import OUT,RUN,bindings,predict,source_file,fit_core,fit_head
from skin_crossfit_correction import StableColorModel,features,ARMS
from skin_support_curve import patient_roles,SkinRepresentation
from skin_pair_train import subset,write
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_neural_reference_verify import numeric_metrics,compare_metrics
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `numpy_head` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_crossfit_correction_verify.py#L19) |
| `main` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_crossfit_correction_verify.py#L27) |
