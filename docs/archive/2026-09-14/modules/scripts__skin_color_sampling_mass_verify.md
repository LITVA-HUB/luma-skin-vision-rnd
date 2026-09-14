# `scripts/skin_color_sampling_mass_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_color_sampling_mass_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Audit person-mass controls independently from their sampler implementation.

SHA-256 исходника: `b858c71f5a28eaf23fa65e8a99d00667c60eba8015b6a8532d31e4d1505feda7`. Строк: **76**.

## Зависимости

```python
import os
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from skin_color_sampling_mass_train import OUT,RUN,bindings
from skin_color_sampling_verify import independent_probability,indices_for,refit
from skin_support_curve_verify import check_summary
from skin_support_curve_train import predict
from skin_support_curve import SkinRepresentation,patient_roles
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_color_sampling_mass_verify.py#L20) |
