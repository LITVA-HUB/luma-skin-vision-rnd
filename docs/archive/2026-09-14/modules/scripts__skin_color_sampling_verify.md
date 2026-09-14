# `scripts/skin_color_sampling_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_color_sampling_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Scalar color geometry, sampling mass identities and exact state audit.

SHA-256 исходника: `b57e0e1ae84c21d03d7a6b89ef1b202475877d8e3e00104655595d41b77442e2`. Строк: **125**.

## Зависимости

```python
import os
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from skin_color_sampling_train import OUT,RUN,bindings
from skin_support_curve_train import predict,novelty
from skin_support_curve import SkinRepresentation,patient_roles
from skin_support_curve_verify import check_summary
from skin_capture_model import MODES
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `independent_probability` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_color_sampling_verify.py#L20) |
| `indices_for` | FunctionDef | См. реализацию | [L38](../../../../scripts/skin_color_sampling_verify.py#L38) |
| `refit` | FunctionDef | См. реализацию | [L45](../../../../scripts/skin_color_sampling_verify.py#L45) |
| `main` | FunctionDef | См. реализацию | [L64](../../../../scripts/skin_color_sampling_verify.py#L64) |
